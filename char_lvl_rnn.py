#importing torch and numpy
import numpy as np
import torch as t
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

with open('generated_text_file/quote.txt', 'r') as f:
    text = f.read()

#reduced text size by around 10 times
text = text[:479000]
print(text[:5])
print(len(text))

#tokenization
chars = tuple(set(text))
print(chars[:1])
print(len(chars))
#text = tuple(text)

#creating dictionaries
#here key is the unique number and chars are the values
int2char = dict(enumerate(chars))
print(int2char)

#chars are keys and numbers are the values
print(int2char.items())
char2int = {ch:ii for ii,ch in int2char.items()}
print(char2int)

#encode the text
encode = np.array([char2int[ch] for ch in text])
print(encode[:100])

#onehotencode
def oneHotEncode(arr):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    lebelEncoder = LabelEncoder()
    arr = labelEncoder.fit_transform(arr)
    oneHot = OneHotEncoder()
    arr = oneHot.fit_transform(arr)
    return arr

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size*seq_length
    #total number of batches we can make
    n_batches = len(arr)//batch_size_total

    #keep enough chars to make full batches
    #ignore the rest chars because it is not going to affect the model much
    arr = arr[:n_batches*batch_size_total]
    print(arr.shape)
    #creates a marix with rows = batch_size and -1 automatically fills up the required number of columns
    arr = arr.reshape((batch_size, -1))
    print('after resize', arr.shape)

    for n in range(0, arr.shape[1], seq_length):
        #input features
        x = arr[:, n:n+seq_length]

        #target, shifted by 1
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:,1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
    
batches = get_batches(encode, 8, 50)
x, y = next(batches)
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])

#Defining the model
device = t.device('cuda' if t.cuda.is_available else 'cpu')

class LSTM(nn.Module):
    def __init__(self, tokens, n_hidden = 256, rnn_layers = 2, drop_prob = 0.5, lr = 0.001):
        super(LSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_hidden = n_hidden
        self.rnn_layers = rnn_layers
        self.lr = lr

        #creating char disconaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch:ii for ii,ch in self.int2char.items()}

        #defining different layers
        #define lstm
        self.lstm = nn.LSTM(len(self.chars),n_hidden, rnn_layers, batch_first = True, dropout = drop_prob)

        #dropout layer
        self.dropout = nn.Dropout(drop_prob)

        #defining a fully connected layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## TODO: put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (device == 'cuda'):
            hidden = (weight.new(self.rnn_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.rnn_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.rnn_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

# array containing all losses at each epoch
# array containing all losses at each epoch
training_losses = []
validation_losses = []

# load a previous saved model
state_dict = t.load('model.pth')
net.load_state_dict(state_dict)

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = t.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(device == 'cuda'):
        net.cuda()
    
    counter = 0
    trn_batch_size_total = 0
    val_batch_size_total = 0
    val_loss_minm = np.Inf
    n_chars = len(net.chars)
    for e in range(epochs):
        trn_loss = 0
        validation_loss = 0
        trn_running_loss = 0
        val_running_loss = 0
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            trn_batch_size_total += batch_size * seq_length
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = t.from_numpy(x), t.from_numpy(y)
            
            if(device == 'cuda'):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            trn_running_loss += loss.item()

            # loss stats
            if counter % print_every == 0:
                with t.no_grad():
                    # Get validation loss
                    val_h = net.init_hidden(batch_size)
                    val_losses = []
                    net.eval()
                    for x, y in get_batches(val_data, batch_size, seq_length):
                        val_batch_size_total += batch_size * seq_length
                        # One-hot encode our data and make them Torch tensors
                        x = one_hot_encode(x, n_chars)
                        x, y = t.from_numpy(x), t.from_numpy(y)
                        
                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])
                        
                        inputs, targets = x, y
                        if(device == 'cuda'):
                            inputs, targets = inputs.cuda(), targets.cuda()

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output, targets.view(batch_size*seq_length))
                        val_running_loss += val_loss.item()
                        val_losses.append(val_loss.item())

                validation_loss = val_running_loss / val_batch_size_total
                validation_losses.append(val_running_loss)
                trn_loss = trn_running_loss/ trn_batch_size_total
                training_losses.append(trn_running_loss)

                net.train() # reset to train mode after iterationg through validation data
                    
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.4f}...".format(loss.item()),
                    "Val Loss: {:.4f}".format(np.mean(val_losses)))
                if val_loss.item() <= val_loss_minm:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    val_loss_minm,val_loss.item()))
                    # change the name, for saving multiple files
                    model_name = 'rnn_20_epoch.net'

                    checkpoint = {'n_hidden': net.n_hidden,
                                'n_layers': net.rnn_layers,
                                'state_dict': net.state_dict(),
                                'tokens': net.chars}

                    with open(model_name, 'wb') as f:
                        t.save(checkpoint, f)
                    val_loss_minm = val_loss.item()

# define and print the net
n_hidden=512
rnn_layers=2

net = LSTM(chars, n_hidden, rnn_layers)
print(net)

batch_size = 128
seq_length = 100
n_epochs = 20 # start smaller if you are just testing initial behavior

# train the model
train(net, encode, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = t.from_numpy(x)
        
        if(device == 'cuda'):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(device == 'cuda'):
            p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h

def sample(net, size, prime='People', top_k=None):
        
    if(device == 'rnn'):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 1000, prime='People', top_k=5))

#plotting graphs
plt.plot(training_losses, label = 'Training loss')
plt.plot(validation_losses, label = 'Validation loss')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.title('Losses')
plt.legend()