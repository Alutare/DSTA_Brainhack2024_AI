import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import noisereduce as nr
import os
import torch.nn.functional as F
from io import BytesIO
import torch.optim as optim

class AudioDataset(Dataset):
    def __init__(self, folder_path, target_length=160000):
        self.target_length = target_length
        self.audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = self.pad_or_truncate_waveform(waveform)
        return waveform, sample_rate

    def pad_or_truncate_waveform(self, waveform):
        if waveform.size(1) < self.target_length:
            pad_amount = self.target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_amount))
        elif waveform.size(1) > self.target_length:
            waveform = waveform[:, :self.target_length]
        return waveform

class ASRManager:
    def __init__(self):
        # Load the pre-trained model and processor for self-supervised learning
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
    def reduce_noise(self, audio_tensor, sample_rate):
        # Convert audio tensor to numpy array
        audio_numpy = audio_tensor.squeeze().numpy()
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(y=audio_numpy, sr=sample_rate)
        
        # Convert back to tensor
        reduced_noise_tensor = torch.tensor(reduced_noise).unsqueeze(0)
        return reduced_noise_tensor

    def preprocess_audio(self, audio_bytes: bytes) -> torch.Tensor:
        # Load audio data from bytes
        audio_tensor, sample_rate = torchaudio.load(BytesIO(audio_bytes))
        
        # Check if sample rate is 16 kHz (expected)
        if sample_rate != 16000:
            # Resample the audio to 16 kHz
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            sample_rate = 16000
        
        # Reduce noise in the audio
        audio_tensor = self.reduce_noise(audio_tensor, sample_rate)
        
        # Preprocess audio data
        input_values = self.processor(audio_tensor.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        
        return input_values

# Instantiate ASRManager
asr_manager = ASRManager()

# Define your training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
)

# Instantiate your dataset class with the folder path to your audio files
dataset = AudioDataset(folder_path='novice', target_length=160000)

# Define the collate function
def collate_fn(batch):
    input_values = [item[0].squeeze() for item in batch]  # Ensure 1D tensor for input values
    sample_rates = [item[1] for item in batch]

    # Find the maximum length in the batch
    max_length = max([x.size(0) for x in input_values])

    # Pad all waveforms to the maximum length
    input_values_padded = torch.stack([F.pad(x, (0, max_length - x.size(0))) for x in input_values])

    return input_values_padded

data_loader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=collate_fn)

optimizer = optim.AdamW(asr_manager.model.parameters(), lr=1e-4)

# Set the model to train mode
asr_manager.model.train()

# Training loop
for epoch in range(training_args.num_train_epochs):
    for batch in data_loader:
        input_values = batch
        attention_mask = (input_values != 0).float()  # Create attention mask

        # Forward pass
        outputs = asr_manager.model(input_values, attention_mask=attention_mask)

        # Perform any necessary backward pass or optimization
        optimizer.zero_grad()
        # No loss computation or backward pass

        # Update parameters
        optimizer.step()

        # Print training progress
        print(f"Epoch: {epoch}, Loss: No loss computed")

