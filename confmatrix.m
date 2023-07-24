%{
C = confusionmat(ground_truth, predicted_label)
cm = confusionchart(C, {'classical'  'pop'}, 'Title', 'music genre classification', 'RowSummary', 'row-normalized')
%}

function [] = confmatrix(ground_truth,predicted_label)
    figure
    C = confusionmat(ground_truth, predicted_label)
    cm = confusionchart(C, {'classical'  'Jazz','Pop'}, 'Title', 'music genre classification', 'RowSummary', 'row-normalized')
