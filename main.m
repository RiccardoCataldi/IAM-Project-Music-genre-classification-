clear;clc
addpath(genpath(pwd))
windowLength = 0.03;
stepLength = 0.01;
%nfft = 1024;

SNR = 5;

[chromaJazzTrain,mfccJazzTrain] = extract_from_path('Jazz/Train/','*mp3', windowLength, stepLength);

[chromaJazzTest,mfccJazzTest] = extract_from_path('Jazz/Test/','*mp3', windowLength, stepLength);

chromaJazz = [chromaJazzTrain chromaJazzTest]; % concatenating train and test

%mfccJazz = [mfccJazzTrain mfccJazzTest];

[chromaPopTrain,mfccPopTrain]= extract_from_path('Pop/Train/','*mp3', windowLength, stepLength);

[chromaPopTest,mfccPopTest] = extract_from_path('Pop/Test/','*mp3', windowLength, stepLength);

chromaPop = [chromaPopTrain chromaPopTest]; % concatenating train and test

%mfccPop = [mfccPopTrain mfccPopTest];

[chromaRockTrain,mfccRockTrain] = extract_from_path('Rock/Train/','*mp3', windowLength, stepLength);

[chromaRockTest,mfccRockTest] = extract_from_path('Rock/Test/','*mp3', windowLength, stepLength);

chromaRock = [chromaRockTrain chromaRockTest]; % concatenating train and test

%mfccRock = [mfccRockTrain mfccRockTest];

%Concatenazione Chroma Train
[labelChromaJazz,labelChromaPop,labelChromaRock] = set_labels(chromaJazzTrain,chromaPopTrain,chromaRockTrain);
all_train_chroma = [chromaJazzTrain chromaPopTrain chromaRockTrain];
all_labels_chroma = [labelChromaJazz;labelChromaPop;labelChromaRock];

%Normalizzazione Chroma Train
[all_train_chroma,mn_train_chroma,st_train_chroma] = normalize_feats(all_train_chroma);

%Concatenazione Mfcc Train
[labelMfccJazz,labelMfccPop,labelMfccRock] = set_labels(mfccJazzTrain,mfccPopTrain,mfccRockTrain);
all_train_mfcc = [mfccJazzTrain mfccPopTrain mfccRockTrain];
all_labels_mfcc = [labelMfccJazz;labelMfccPop;labelMfccRock];

%Normalizzazione Mfcc Train
[all_train_mfcc,mn_all_mfcc,st_all_mfcc] = normalize_feats(all_train_mfcc);

% concatenating chromogram+mfcc Train

allFeatsJazzTrain = [chromaJazzTrain; mfccJazzTrain];

allFeatsPopTrain = [chromaPopTrain; mfccPopTrain];

allFeatsRockTrain = [chromaRockTrain; mfccRockTrain];

[labelallFeatsJazz,labelallFeatsPop,labelallFeatsRock] = set_labels(allFeatsJazzTrain,allFeatsPopTrain,allFeatsRockTrain);

allFeatsTrain = [allFeatsJazzTrain allFeatsPopTrain allFeatsRockTrain];

allFeatsLabels = [labelallFeatsJazz; labelallFeatsPop; labelallFeatsRock];

% normalize chromogram+mfcc

[allFeatsTrain,mn_feats_train,st_feats_train] = normalize_feats(allFeatsTrain);

%concatenazione chroma Test

[labelChromaJazzTest,labeChromaPopTest,labelChromaRockTest] = set_labels(chromaJazzTest,chromaPopTest,chromaRockTest);

all_test_chroma = [chromaJazzTest chromaPopTest chromaRockTest];
ground_truth_chroma = [labelChromaJazzTest;labeChromaPopTest;labelChromaRockTest];

% normalize the test set
all_test_chroma = normalize_featsTest(all_test_chroma,mn_train_chroma,st_train_chroma);

%concatenazione mfcc Test

[labelMfccJazzTest,labelMfccPopTest,labelMfccRockTest] = set_labels(mfccJazzTest,mfccPopTest,mfccRockTest);
all_test_mfcc = [mfccJazzTest mfccPopTest mfccRockTest];
ground_truth_mfcc = [labelMfccJazz;labelMfccPopTest;labelMfccRockTest];

% normalize the test set
all_test_mfcc = normalize_featsTest(all_test_mfcc,mn_all_mfcc,st_all_mfcc);

% concatenating chromogram+MFCCs test

allFeatsJazzTest = [chromaJazzTest; mfccJazzTest];
allFeatsPopTest = [chromaPopTest; mfccPopTest];
allFeatsRockTest = [chromaRockTest; mfccRockTest];

allFeatsTest = [allFeatsJazzTest allFeatsPopTest allFeatsRockTest];

[labelAllFeatsJazzTest,labelAllFeatsPopTest,labelAllFeatsRockTest] = set_labels(allFeatsJazzTest,allFeatsPopTest,allFeatsRockTest);

ground_truth_allFeats = [labelAllFeatsJazzTest; labelAllFeatsPopTest; labelAllFeatsRockTest]; % contains the real labels

[allFeatsTest] = normalize_featsTest(allFeatsTest,mn_feats_train,st_feats_train);

k = [1 10 20];

%KNN Chroma

disp('----------KNNChroma----------------')

%function [predicted_label, rate] = kNN(k, features1, label1, features2, label2,idx)

[plabel_chroma,rateChroma] = kNN(k,all_train_chroma,all_labels_chroma,all_test_chroma,ground_truth_chroma);

%KNN Mfcc
disp('----------KNNMfcc----------------')
[plabel_mfcc,rateMfcc] = kNN(k,all_train_mfcc,all_labels_mfcc,all_test_mfcc,ground_truth_mfcc);

%KNN All
disp('----------KNNAll----------------')
[plabel_All,rateAll,MdlKnnAll] = kNN(k,allFeatsTrain,allFeatsLabels,allFeatsTest,ground_truth_allFeats);

%adding babble noise Jazz

add_noise('babble.wav','Jazz/Test/','*mp3',SNR,'Jazz_test_noise/Test/')

%adding babble noise Pop

add_noise('babble.wav','Pop/Test/','*mp3',SNR,'Pop_test_noise/Test/')

%adding babble noise Rock

add_noise('babble.wav','Rock/Test/','*mp3',SNR,'Rock_test_noise/Test/')


%KNN Babble All

[chromaJazzBabbleTest,mfccJazzBabbleTest] = extract_from_path('Jazz_test_noise/Test/','*wav', windowLength, stepLength);
[chromaPopBabbleTest,mfccPopBabbleTest] = extract_from_path('Pop_test_noise/Test/','*wav', windowLength, stepLength);
[chromaRockBabbleTest,mfccRockBabbleTest] = extract_from_path('Rock_test_noise/Test/','*wav', windowLength, stepLength);


% concatenating chromogram+mfcc Test

allFeatsJazzBabbleTest = [chromaJazzBabbleTest; mfccJazzBabbleTest];

allFeatsPopBabbleTest = [chromaPopBabbleTest; mfccPopBabbleTest];

allFeatsRockBabbleTest = [chromaRockBabbleTest; mfccRockBabbleTest];

[labelallFeatsJazzBabble,labelallFeatsPopBabble,labelallFeatsRockBabble] = set_labels(allFeatsJazzBabbleTest,allFeatsPopBabbleTest,allFeatsRockBabbleTest);

allFeatsTestBabble = [allFeatsJazzBabbleTest allFeatsPopBabbleTest allFeatsRockBabbleTest];


ground_truth_allFeatsBabble = [labelallFeatsJazzBabble; labelallFeatsPopBabble; labelallFeatsRockBabble];

% normalize chromogram+mfcc

allFeatsTestBabble = normalize_featsTest(allFeatsTestBabble,mn_feats_train,st_feats_train);

%KNN Babble All
disp('----------KNNBabble----------------')
[predLAllBabble, rateAllBabble] = predict_best_kNN(10,MdlKnnAll,allFeatsTestBabble,ground_truth_allFeatsBabble)

%% --- SPECTRAL SUBSTRACTION / KNN5 ---
% --- sub noise ---
disp('.......... denoising the test-noisyFiles and saving in the new enhanced-paths ..........')

sub_noise('Jazz_test_noise/Test/', '*wav', 'Jazz_enh_test/') % denoising and saving denoised jazz in the new enhanced-path
sub_noise('Pop_test_noise/Test/', '*wav', 'Pop_enh_test/') % denoising and saving denoised jazz in the new enhanced-path
sub_noise('Rock_test_noise/Test/', '*wav', 'Rock_enh_test/') % denoising and saving denoised jazz in the new enhanced-path

% --- MFCCs kNN 5 (applied to enhanced-files, in the enhanced-paths) ---
% extract mfccs from directories' enhanced-paths (metal, rap, rock80's)
disp('extracting features from enhanced files...')
disp(' ')

%KNN Babble All

[chromaJazzEnhTest,mfccJazzEnhTest] = extract_from_path('Jazz_enh_test/','*wav', windowLength, stepLength);
[chromaPopEnhTest,mfccPopEnhTest] = extract_from_path('Pop_enh_test/','*wav', windowLength, stepLength);
[chromaRockEnhTest,mfccRockEnhTest] = extract_from_path('Rock_enh_test/','*wav', windowLength, stepLength);


% concatenating chromogram+mfcc Test

allFeatsJazzEnhTest = [chromaJazzEnhTest; mfccJazzEnhTest];

allFeatsPopEnhTest = [chromaPopEnhTest; mfccPopEnhTest];

allFeatsRockEnhTest = [chromaRockEnhTest; mfccRockEnhTest];

[labelallFeatsJazzEnh,labelallFeatsPopEnh,labelallFeatsRockEnh] = set_labels(allFeatsJazzEnhTest,allFeatsPopEnhTest,allFeatsRockEnhTest);

allFeatsTestEnh = [allFeatsJazzEnhTest allFeatsPopEnhTest allFeatsRockEnhTest];


ground_truth_allFeatsEnh = [labelallFeatsJazzEnh; labelallFeatsPopEnh; labelallFeatsRockEnh];

% normalize chromogram+mfcc

allFeatsTestEnh = normalize_featsTest(allFeatsTestEnh,mn_feats_train,st_feats_train);


%KNN Enh All
disp('----------KNNEnh----------------')
[predlAllEnh, ratelAllEnh]=predict_best_kNN(10,MdlKnnAll,allFeatsTestEnh,ground_truth_allFeatsEnh);

%{
%confusion matrix chroma
confmatrix(plabel_chroma,ground_truth_chroma)


%confusion matrix mfcc
confmatrix(plabel_mfcc,ground_truth_mfcc)


%confusion matrix all
confmatrix(plabel_All,ground_truth_allFeats)


%confusion matrix noise (ground_truth_allFeatsBabble)
confmatrix(predLAllBabble,ground_truth_allFeatsBabble)


%confusion matrix enh
confmatrix(predlAllEnh,ground_truth_allFeatsEnh)
%}

%% --- CONFUSION MATRIX ---
disp('.......... realizing confusion matrixes ..........')
disp(' ')

% --- chromagram ---
disp('1st confusion matrix: chromagram ...')
realize_confusion_matrix(plabel_chroma,ground_truth_chroma, 'Confusion matrix: CHROMA') % function for confusion matrix

% --- mfccs ---
disp('2nd confusion matrix: mfccs ...')
realize_confusion_matrix(plabel_mfcc,ground_truth_mfcc, 'Confusion matrix: MFCCs') % function for confusion matrix

% --- chromagram + mfccs ---
disp('3rd confusion matrix: chromagram + mfccs ...')
realize_confusion_matrix(plabel_All,ground_truth_allFeats, 'Confusion matrix: CHROMA + MFCCs') % function for confusion matrix

% --- mfccs noisy ---
disp('4th confusion matrix: All noisy ...')
realize_confusion_matrix(predLAllBabble,ground_truth_allFeatsBabble, 'Confusion matrix: All noisy') % function for confusion matrix

% --- mfccs enhanced ---
disp('5th confusion matrix: All enhanced ...')
realize_confusion_matrix(predlAllEnh,ground_truth_allFeatsEnh, 'Confusion matrix: All enhanced') % function for confusion matrix








