# IAM-Project-Music-genre-classification-
Project IAM 2022/2023 Riccardo Cataldi

For my project, I have selected 18 songs, divided into three genres: Jazz, Pop, and Rock. Half of the songs are used for the training phase, and the other half for the testing phase. 
Tests are also performed by adding noise to the songs and then reducing the noise using specific functions. The feature extraction is carried out using two functions:
MFCC (Mel Frequency Cepstral Coefficients) extraction and Chroma feature extraction.

The next step is the concatenation and normalization of the features. This process is performed for both the training and testing data. Subsequently, using three different values for k (the number of neighbors), kNN functions take the extracted features from the test and training data. kNN is first used only with Chroma features, then only with MFCC features, and finally with both.

Once the program is executed, it shows that the best recognition rate for kNN is obtained with 10 neighbors (k) using both features (MFCC and Chroma).

Feature Extraction:
Chroma and MFCC features are extracted from the training and testing songs for the music genres Jazz, Pop, and Rock. The functions used for feature extraction are "extract_from_path," which returns the matrices of Chroma and MFCC features for each genre.

Preparation of Training and Testing Data:
The Chroma and MFCC features are concatenated for the training and testing data of each genre. A vector of labels corresponding to the music genres is also created for each set of features.

Normalization:
The features are normalized for the training data using the function "normalize_feats." This process helps balance the features and ensures they have similar importance during classification.

Classification with k-NN:
Classification is performed using the k-NN (k-Nearest Neighbors) algorithm for Chroma features, MFCC features, and a combination of both. The recognition percentage is calculated for each k configuration, and the classification results are returned.

Adding Noise and Denoising:
Noise is added to the test audio files using the file "babble.wav," and a denoising algorithm known as spectral subtraction is applied to remove the added noise. Chroma and MFCC features are then extracted from the denoised audio files.

Classification with Noisy and Denoised Data:
Classification is performed using k-NN on the noisy and denoised data, and the classification results are returned.

Confusion Matrices:
Confusion matrices are created to evaluate the performance of the classification model on different datasets and conditions (Chroma, MFCC, noisy, denoised).

    
 
