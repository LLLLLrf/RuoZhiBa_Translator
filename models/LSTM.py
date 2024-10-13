import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
    def fit(self, X_train, y_train, epochs, batch_size, validation_data):
        X_val, y_val = validation_data
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
                
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
            val_loss = self.evaluate(X_val, y_val)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
    
    def evaluate(self, X_val, y_val):
        with torch.no_grad():
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_pred = self(X_val)
            loss = F.mse_loss(y_pred, y_val)
        return loss.item()
    
    def predict(self, X_test):
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_pred = self(X_test)
        return y_pred.numpy()
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
