# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

### Dataset Name: ner_dataset.csv

Size: Contains thousands of words grouped into sentences with entity annotations.

#### Columns:

Sentence # – Sentence ID

Word – Individual word/token in the sentence

POS – Part-of-speech tag

Tag – Named entity tag (e.g., O, B-PER, I-LOC, etc.)


## DESIGN STEPS

STEP 1:
Import necessary libraries.

STEP 2:
Load dataset , Read and clean the input data.

STEP 3:
Structure data into sentences with word-tag pairs.

STEP 4:
Convert words and tags to indices using vocab dictionaries.

STEP 5:
Pad sequences, convert to tensors, and batch them.

STEP 6:
Create a model with Embedding, BiLSTM, and Linear layers.

STEP 7:
Use training data to update model weights with loss and optimizer.

STEP 8:
Check performance on validation data after each epoch.

STEP 9:
Display predictions or plot loss curves.

## PROGRAM
### Name: S.NAVADEEP
### Register Number: 212224230180
```python
class BiLSTMTagger(nn.Module):
  def __init__(self, vocab_size, tagset_size, embedding_dim = 50, hidden_dim = 100):
    super(BiLSTMTagger, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.dropout = nn.Dropout(0,1)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(hidden_dim * 2, tagset_size)

  def forward(self, x):
    x = self.embedding(x)
    x = self.dropout(x)
    x, _ = self.lstm(x)
    return self.fc(x)      

model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")          

    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="739" height="627" alt="image" src="https://github.com/user-attachments/assets/70b42b6c-69f4-4b11-a410-e5bc490b11f2" />

### Sample Text Prediction

<img width="447" height="554" alt="image" src="https://github.com/user-attachments/assets/c116f7b0-8aa2-4d16-80d1-508b9a03099f" />

## RESULT

Thus the LSTM-based Named Entity Recognition (NER) model was successfully developed and trained.

