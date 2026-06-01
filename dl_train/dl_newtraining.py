import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import DL_training as DL

# hyperparameter settings
CNN_layers = 1
LSTM_layers = 1
pool_size_param = 2
learning_rate_param = 0.0005
batch_param = 1000
dropout_percent = 0.10
filters_param = 50
mem_cells = 50
mem_cells2 = 10
kernel_size_param = 12
epoch_param = 1500
initializer_param = "lecun_normal"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training on: {device} ---")

class testClassifier(nn.Module):
    def __init__(self, input_dim = 384, hidden_dim = 128, output_dim = 4):
        super(testClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x=self.dropout(x)
        return self.net(x)

model = testClassifier(input_dim=DL.ts_len).to(device)

training_data = torch.tensor(DL.train, dtype=torch.float32).view(len(DL.train), -1)
training_labels = torch.tensor(DL.train_target, dtype=torch.long)

test_data = torch.tensor(DL.test, dtype=torch.float32).view(len(DL.test), -1)
test_labels = torch.tensor(DL.test_target, dtype=torch.long)

train_dataset = TensorDataset(training_data, training_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_param, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_param, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_test_loss = float('inf')
patience = 10  #wait for improvement before giving up
epochs_no_improve = 0

print("--- Starting PyTorch Training Loop ---")

for epoch in range(epoch_param):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
        correct_predictions += (outputs.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_X.size(0)


        if epoch_loss < best_test_loss:
                best_test_loss = epoch_loss
                epochs_no_improve = 0
                # Optional but highly recommended: Save the best model state!
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            print(f"Notice: No improvement in test loss for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= patience:
            print(f"\n--- EARLY STOPPING TRIGGERED AT EPOCH {epoch} ---")
            print("The model has started overfitting. Stopping training.")
            break

    #average loss and accuracy
    avg_loss = epoch_loss / total_samples
    avg_acc = correct_predictions / total_samples
    
    #Print every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epoch_param} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

#evaluate on test set
model.eval()
with torch.no_grad():
    # Send test data to GPU
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    test_outputs = model(test_data)
    test_loss = criterion(test_outputs, test_labels)
    test_accuracy = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
    
    print("\n--- Final Results ---")
    print(f"Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}")


# class PositionalEncoding(Layer): #class for positional encoding layer for transformer model C.H
#     def __init__(self, input_dim = 500, hidden_dim = 128, output_dim = 4, **kwargs):
#         super(PositionalEncoding, self).__init__(**kwargs)
#         self.sequence_length = sequence_length
#         self.d_model = d_model
#         self.pos_encoding = self.positional_encoding(sequence_length, d_model)

#     def get_angles(self, pos, i, d_model):
#         angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
#         return pos * angles

#     def positional_encoding(self, sequence_length, d_model):
#         angle_rads = self.get_angles(
#             pos=tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis],
#             i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
#             d_model=d_model
#         )
#         sines = tf.math.sin(angle_rads[:, 0::2])
#         cosines = tf.math.cos(angle_rads[:, 1::2])
#         pos_encoding = tf.reshape(
#             tf.concat([sines[..., tf.newaxis], cosines[..., tf.newaxis]], axis=-1),
#             [sequence_length, d_model]
#         )
#         return tf.cast(pos_encoding[tf.newaxis, ...], tf.float32)

#     def call(self, inputs):
#         return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

#     def get_config(self):
#         config = super().get_config()
#         config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
#         return config



# inputs = Input(shape=(DL.ts_len, 1))

# # 1. Linear Projection / Embedding
# x = Conv1D(
#     filters=DL.filters_param, 
#     kernel_size=DL.kernel_size_param, 
#     activation="relu", 
#     padding="same",
#     kernel_initializer=DL.initializer_param
# )(inputs)

# # 2. Positional Encoding, which is added to the input of the transformer blocks to provide information about the position of each element in the sequence. C.H
# x = PositionalEncoding(sequence_length=DL.ts_len, d_model=DL.filters_param)(x)

# # 3. Transformer Blocks: Each block consists of multi-head self-attention followed by a feed-forward network, with skip connections and layer normalization. C.H
# num_heads = 4       
# ff_dim = 128        
# num_blocks = 2      

# for _ in range(num_blocks):
#     # Multi-Head Self Attention
#     attn_output = MultiHeadAttention(key_dim=DL.filters_param, num_heads=num_heads)(x, x)
#     attn_output = Dropout(DL.dropout_percent)(attn_output)
    
#     # Add & Norm
#     out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))
    
#     # Feed Forward Network
#     ff_output = Dense(ff_dim, activation="relu", kernel_initializer=DL.initializer_param)(out1)
#     ff_output = Dropout(DL.dropout_percent)(ff_output)
#     ff_output = Dense(DL.filters_param, kernel_initializer=DL.initializer_param)(ff_output)
    
#     # Add & Norm
#     x = LayerNormalization(epsilon=1e-6)(Add()([out1, ff_output]))

# # 4. Pooling and Final Output
# x = GlobalAveragePooling1D()(x)
# x = Dropout(DL.dropout_percent)(x)
# outputs = Dense(4, activation="softmax", kernel_initializer=DL.initializer_param)(x)

# # Build the model
# model = Model(inputs=inputs, outputs=outputs)

# # name for output pickle file containing model info
# model_name = "best_model_{}_{}_length{}.keras".format(DL.kk, DL.model_type, DL.ts_len)




#new classifier training code using PyTorch

