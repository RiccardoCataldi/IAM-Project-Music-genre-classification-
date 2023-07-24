function [] = realize_confusion_matrix(predicted_label,ground_truth, title)

% using the corresponding function
figure
C = confusionmat(ground_truth, predicted_label);
cm = confusionchart(C, {'Jazz'  'Pop'  'Rock'}, 'Title', title, 'RowSummary', 'row-normalized'); 