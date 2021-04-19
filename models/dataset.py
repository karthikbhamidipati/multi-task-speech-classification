from torch.utils.data import Dataset, DataLoader


class CommonVoiceDataset(Dataset):
    def __init__(self, dataset, dataset_type, feature_type):
        self.accent_mappings = dataset['mappings']['accent']
        self.gender_mappings = dataset['mappings']['gender']
        self.age_mappings = dataset['mappings']['age']
        self.data = dataset['processed_data'][dataset_type]
        self.feature_type = feature_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features_labels_dict = self.data[idx]
        features = features_labels_dict[self.feature_type]
        gender_label = self.gender_mappings['gender2idx'][features_labels_dict['gender']]
        accent_label = self.accent_mappings['accent2idx'][features_labels_dict['accent']]
        return features, (gender_label, accent_label)


def get_data_loaders(data, config):
    # Load train data
    print('Reading Train data', flush=True)
    train_dataset = CommonVoiceDataset(data, 'train_set', config.feature_type)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    print('Reading Validation data', flush=True)
    val_dataset = CommonVoiceDataset(data, 'val_set', config.feature_type)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Load test data
    print('Reading Test data', flush=True)
    test_dataset = CommonVoiceDataset(data, 'test_set', config.feature_type)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
