import re
import numpy as np
import keras

#Fixed parameters
PROCESSING_FIXED = {'start_char': 'G', 
                    'end_char': 'E', 
                    'pad_char': 'A'}

INDICES_TOKEN = {0: 'c',
                 1: 'C',
                 2: '(',
                 3: ')',
                 4: 'O',
                 5: '1',
                 6: '2',
                 7: '=',
                 8: 'N',
                 9: '@',
                 10: '[',
                 11: ']',
                 12: 'n',
                 13: '3',
                 14: 'H',
                 15: 'F',
                 16: '4',
                 17: '-',
                 18: 'S',
                 19: 'Cl',
                 20: '/',
                 21: 's',
                 22: 'o',
                 23: '5',
                 24: '+',
                 25: '#',
                 26: '\\',
                 27: 'Br',
                 28: 'P',
                 29: '6',
                 30: 'I',
                 31: '7',
                 32: PROCESSING_FIXED['start_char'],
                 33: PROCESSING_FIXED['end_char'],
                 34: PROCESSING_FIXED['pad_char']}                
TOKEN_INDICES = {v: k for k, v in INDICES_TOKEN.items()}

def smi_tokenizer(smi):
    """
    Tokenize a SMILES
    """
    pattern =  "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens        

class onehotencoder():
    def __init__(self, max_chars=90):
        # here we define the class variables that will be used in the functions. If we want to use them in the class, we need to use self. in front of the variable name
        # then when we want to use the class variable in the function, we pass self as the first argument to the function and we can access the class variables
        self.max_chars = max_chars
    def smiles_to_num(self, smiles):
        tokens = smi_tokenizer(smiles)
        tokens = [PROCESSING_FIXED['start_char']] + tokens +[PROCESSING_FIXED['end_char']]
        tokens+= [PROCESSING_FIXED['pad_char']]* (self.max_chars-len(tokens))
        num_tokens = [TOKEN_INDICES[t] for t in tokens]
        #print(tokens)
        #create numbers
        return np.asarray(num_tokens) #Return the numbers as a numpy array
    def num_to_onehot(self,tokens):
        onehot = np.zeros((len(tokens), len(INDICES_TOKEN)))
        for i,token in enumerate(tokens):
            onehot[i, TOKEN_INDICES[token]] = 1
        return onehot
        #This function will convert the numbers to one hot encoding. It should take a list of numbers and return a list of one hot encoded vectors with the shape (max_chars, 35)
    def generate_data(self, smiles):
        nums = self.smiles_to_num(smiles)
        data = self.num_to_onehot(nums)
        return np.asarray(data)
        #This function should take a list of smiles strings and return a numpy array of one hot encoded vectors with the shape (number of smiles, max_chars, 35)
        #You can use the smiles_to_num and num_to_onehot functions to help you with this
        #This function will be used to generate the training and validation data

        
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size, data, max_len, num_chars, use_multiprocessing=True, workers=6, shuffle=True):
        super().__init__(use_multiprocessing=use_multiprocessing, workers=workers)
        'Initialization'
        encoder = onehotencoder(max_chars=max_len)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data = data
        self.num_chars = num_chars
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.max_len-1, self.num_chars))
        y = np.empty((self.batch_size, self.max_len-1, self.num_chars))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            smiles = self.data[ID]
            one_hot_smi = self.encoder.generate_data(smiles)
            X[i] = one_hot_smi[:-1]
            y[i] = one_hot_smi[1:]
        return X, y
    #You can give this a try with the tutorial and see if you can get it to work https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    #This will be used to generate the data for the model. It should take a list of smiles strings and the corresponding labels and return the one hot encoded vectors in batches
    #You can use the onehotencoder class to help you with this.
    #It is important to follow the function signature of the keras.utils.Sequence class so that it can be used with the model.fit_generator function
    #This is al detailed in the tutorial above