import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_glove_vecs(glove_file):
    """
    Reads GloVe word vectors from a file and creates mappings for words to indices, 
    indices to words, and words to their corresponding vector representations.

    Args:
        glove_file (str): Path to the GloVe embedding file.

    Returns:
        tuple: A tuple containing:
            - words_to_index (dict): A dictionary mapping words to unique integer indices.
            - index_to_words (dict): A dictionary mapping integer indices to words.
            - word_to_vec_map (dict): A dictionary mapping words to their corresponding 
              GloVe vector representations as NumPy arrays.

    The function:
    - Reads the GloVe file line by line.
    - Extracts words and their corresponding vector representations.
    - Stores word-to-vector mappings in `word_to_vec_map`.
    - Assigns unique indices to words in `words_to_index` and `index_to_words`.

    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_csv(filename = 'data/emojify_data.csv'):
    """
     Reads a CSV file containing text phrases and their corresponding emoji labels.

    Args:
        filename (str, optional): Path to the CSV file. Defaults to 'data/emojify_data.csv'.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): An array of text phrases.
            - Y (numpy.ndarray): An array of corresponding emoji labels as integers.

    The function:
    - Reads the CSV file line by line.
    - Extracts text phrases and their associated emoji labels.
    - Converts them into NumPy arrays for further processing.

    """
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
              
    
def print_predictions(X, pred):
    """
    Prints text phrases alongside their predicted emoji labels.

    Args:
        X (numpy.ndarray): An array of text phrases.
        pred (numpy.ndarray or list): An array or list of predicted emoji labels (as integers).

    The function:
    - Iterates through the given text phrases.
    - Converts predicted integer labels into their corresponding emojis using `label_to_emoji()`.
    - Prints each phrase along with its predicted emoji.

    """
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        
        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    """
    This function prints and plots the confusion matrix. 
    Arguments:
        y_actu -- true labels
        y_pred -- predicted labels
        title -- title of the plot
        cmap -- color map
    Returns:
        None
    """
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
    
    
def predict(X, Y, W, b, word_to_vec_map):
    """
    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))
    any_word = list(word_to_vec_map.keys())[0]
    # number of classes  
    n_h = word_to_vec_map[any_word].shape[0] 
    
    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        words = X[j].lower().split()
        
        # Average words' vectors
        avg = np.zeros((n_h,))
        count = 0
        for w in words:
            if w in word_to_vec_map:
                avg += word_to_vec_map[w]
                count += 1
        
        if count > 0:
            avg = avg / count

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred