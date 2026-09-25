function surrogate_accuracy_benchmark()

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultFigureColor', 'w');
set(0, 'DefaultAxesFontAngle', 'normal');
set(0, 'DefaultTextFontAngle', 'normal');
try
    set(0, 'DefaultAxesXTickLabelRotation', 0);
    set(0, 'DefaultAxesYTickLabelRotation', 0);
catch
end

outputDirectoryName = 'benchmark_output';
currentWorkingDirectory = pwd;
outputDirectoryPath = fullfile(currentWorkingDirectory, outputDirectoryName);
directoryExistenceFlag = exist(outputDirectoryPath, 'dir');
if directoryExistenceFlag ~= 7
    [mkStatus, mkMessage] = mkdir(outputDirectoryPath);
    if mkStatus == 0
        error('directory creation failed: %s', mkMessage);
    end
end

figurePositionValue = [100, 100, 1500, 500];
titleFontSizeValue = 11;
labelFontSizeValue = 10;
tickFontSizeValue = 9;
legendFontSizeValue = 8;

colorGrayValue = [0.45, 0.45, 0.45];
colorBlueValue = [0.20, 0.45, 0.70];
colorRedValue = [0.80, 0.25, 0.20];
colorLightGrayValue = [0.82, 0.82, 0.82];
colorPalette = {colorGrayValue, colorBlueValue, colorRedValue};

fileNamesList = {'data_random.mat', 'data_lhs.mat', 'data_dynamic.mat'};
datasetNamesList = {'Random', 'LHS', 'Dynamic Spatial'};
numberOfDatasets = numel(fileNamesList);

loadedDatasetContainer = cell(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    currentFileName = fileNamesList{datasetIndex};
    fileExistsFlag = exist(currentFileName, 'file');
    if fileExistsFlag == 2
        currentLoadedStruct = load(currentFileName);
        loadedDatasetContainer{datasetIndex} = currentLoadedStruct;
    else
        error('file not found: %s', currentFileName);
    end
end

datasetVisualizationRoutine(loadedDatasetContainer, datasetNamesList, ...
    colorPalette, colorLightGrayValue, figurePositionValue, ...
    titleFontSizeValue, labelFontSizeValue, tickFontSizeValue, ...
    legendFontSizeValue, outputDirectoryPath);

surrogateConfigurationStructure.treeCountValue = 100;
surrogateConfigurationStructure.minimumLeafSizeValue = 1;
surrogateConfigurationStructure.maximumSplitCountValue = 127;
surrogateConfigurationStructure.criticalWeightFactorValue = 50;

resultContainerStructure = struct();
for datasetIndex = 1:numberOfDatasets
    currentLoadedStruct = loadedDatasetContainer{datasetIndex};

    trainingFeatureMatrix = currentLoadedStruct.X;
    trainingTargetVector = currentLoadedStruct.y;
    thresholdValue = currentLoadedStruct.Rthr;
    testingFeatureMatrix = currentLoadedStruct.X_test;
    testingTargetVector = currentLoadedStruct.y_test;

    clockHandle = tic;
    trainedForestStructure = build_weighted_random_forest(trainingFeatureMatrix, ...
        trainingTargetVector, thresholdValue, surrogateConfigurationStructure);
    elapsedTrainingTime = toc(clockHandle);

    clockHandle = tic;
    predictedVector = evaluate_random_forest(trainedForestStructure, testingFeatureMatrix);
    elapsedPredictionTime = toc(clockHandle);

    isCriticalTestVector = determine_critical_indicator(testingTargetVector, thresholdValue);
    criticalPredictedValues = extract_masked_values(predictedVector, isCriticalTestVector);
    criticalTargetValues = extract_masked_values(testingTargetVector, isCriticalTestVector);

    residualVector = criticalPredictedValues - criticalTargetValues;
    squaredResidualVector = residualVector .^ 2;
    meanSquaredResidualValue = compute_scalar_mean(squaredResidualVector);
    rmseCriticalValue = sqrt(meanSquaredResidualValue);

    absoluteResidualVector = abs(residualVector);
    maeCriticalValue = compute_scalar_mean(absoluteResidualVector);

    predictedClassVector = determine_critical_indicator(predictedVector, thresholdValue);
    truePositiveCount = compute_logical_and_count(predictedClassVector, isCriticalTestVector);
    falsePositiveCount = compute_logical_and_count(predictedClassVector, ~isCriticalTestVector);
    falseNegativeCount = compute_logical_and_count(~predictedClassVector, isCriticalTestVector);

    precisionDenominator = truePositiveCount + falsePositiveCount;
    if precisionDenominator < 1
        precisionDenominator = 1;
    end
    precisionValue = truePositiveCount / precisionDenominator;

    recallDenominator = truePositiveCount + falseNegativeCount;
    if recallDenominator < 1
        recallDenominator = 1;
    end
    recallValue = truePositiveCount / recallDenominator;

    f1Denominator = precisionValue + recallValue;
    if f1Denominator < 1e-12
        f1Denominator = 1e-12;
    end
    f1Value = 2 * precisionValue * recallValue / f1Denominator;

    resultContainerStructure(datasetIndex).datasetLabel = datasetNamesList{datasetIndex};
    resultContainerStructure(datasetIndex).criticalRMSEValue = rmseCriticalValue;
    resultContainerStructure(datasetIndex).criticalMAEValue = maeCriticalValue;
    resultContainerStructure(datasetIndex).precisionValue = precisionValue;
    resultContainerStructure(datasetIndex).recallValue = recallValue;
    resultContainerStructure(datasetIndex).f1ScoreValue = f1Value;
    resultContainerStructure(datasetIndex).criticalCountValue = currentLoadedStruct.crit_found;
    resultContainerStructure(datasetIndex).trainingDuration = elapsedTrainingTime;
    resultContainerStructure(datasetIndex).predictionDuration = elapsedPredictionTime;
end

resultVisualizationRoutine(resultContainerStructure, datasetNamesList, ...
    colorPalette, colorLightGrayValue, figurePositionValue, ...
    titleFontSizeValue, labelFontSizeValue, tickFontSizeValue, ...
    legendFontSizeValue, outputDirectoryPath);

end


function datasetVisualizationRoutine(loadedDatasetContainer, datasetNamesList, ...
    colorPalette, colorLightGrayValue, figurePositionValue, ...
    titleFontSizeValue, labelFontSizeValue, tickFontSizeValue, ...
    legendFontSizeValue, outputDirectoryPath)

numberOfDatasets = numel(loadedDatasetContainer);

figureHandle = figure('Position', figurePositionValue, 'Color', 'w');

subplot(2, 3, 1);
hold on;
for datasetIndex = 1:numberOfDatasets
    currentStruct = loadedDatasetContainer{datasetIndex};
    currentTargetVector = currentStruct.y;
    currentDisplayName = datasetNamesList{datasetIndex};
    currentColorValue = colorPalette{datasetIndex};
    histogram(currentTargetVector, 30, 'Normalization', 'pdf', ...
        'FaceColor', currentColorValue, 'FaceAlpha', 0.5, ...
        'DisplayName', currentDisplayName);
end
firstThresholdValue = loadedDatasetContainer{1}.Rthr;
xline(firstThresholdValue, 'k--', 'LineWidth', 1.8, 'DisplayName', 'Threshold');
xlabel('y', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
ylabel('Density', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Distribution of y-values', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
legend('Location', 'best', 'FontSize', legendFontSizeValue);
set(gca, 'FontSize', tickFontSizeValue, 'FontAngle', 'normal', 'XTickLabelRotation', 0);
grid on; box on; xtickangle(0); hold off;

subplot(2, 3, 2);
criticalFractionValues = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    currentStruct = loadedDatasetContainer{datasetIndex};
    currentTargetVector = currentStruct.y;
    currentThresholdValue = currentStruct.Rthr;
    indicatorVector = currentTargetVector <= currentThresholdValue;
    indicatorMean = compute_scalar_mean(double(indicatorVector));
    criticalFractionValues(datasetIndex) = 100 * indicatorMean;
end
barHandle = bar(criticalFractionValues, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
for datasetIndex = 1:numberOfDatasets
    barHandle.CData(datasetIndex, :) = colorPalette{datasetIndex};
end
set(gca, 'XTickLabel', datasetNamesList, 'FontSize', tickFontSizeValue, ...
    'FontAngle', 'normal', 'XTickLabelRotation', 0);
ylabel('Fraction (%)', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Critical-point fraction', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
maxFractionValue = compute_scalar_maximum(criticalFractionValues);
for datasetIndex = 1:numberOfDatasets
    offsetValue = maxFractionValue * 0.03;
    labelYPosition = criticalFractionValues(datasetIndex) + offsetValue;
    labelText = sprintf('%.2f%%', criticalFractionValues(datasetIndex));
    text(datasetIndex, labelYPosition, labelText, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end

subplot(2, 3, 3);
criticalCountValues = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    currentStruct = loadedDatasetContainer{datasetIndex};
    currentTargetVector = currentStruct.y;
    currentThresholdValue = currentStruct.Rthr;
    indicatorVector = currentTargetVector <= currentThresholdValue;
    indicatorSum = compute_logical_sum(indicatorVector);
    criticalCountValues(datasetIndex) = indicatorSum;
end
barHandle = bar(criticalCountValues, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
for datasetIndex = 1:numberOfDatasets
    barHandle.CData(datasetIndex, :) = colorPalette{datasetIndex};
end
set(gca, 'XTickLabel', datasetNamesList, 'FontSize', tickFontSizeValue, ...
    'FontAngle', 'normal', 'XTickLabelRotation', 0);
ylabel('Count', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Critical-point count', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
maxCountValue = compute_scalar_maximum(criticalCountValues);
for datasetIndex = 1:numberOfDatasets
    offsetValue = maxCountValue * 0.03;
    labelYPosition = criticalCountValues(datasetIndex) + offsetValue;
    labelText = sprintf('%d', criticalCountValues(datasetIndex));
    text(datasetIndex, labelYPosition, labelText, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end

concatenatedFeatureMatrix = concatenate_feature_matrices(loadedDatasetContainer);
centeredFeatureMatrix = center_feature_matrix(concatenatedFeatureMatrix);
principalDirections = extract_principal_directions(centeredFeatureMatrix);

for datasetIndex = 1:numberOfDatasets
    subplot(2, 3, 3 + datasetIndex);
    currentStruct = loadedDatasetContainer{datasetIndex};
    currentFeatureMatrix = currentStruct.X;
    currentTargetVector = currentStruct.y;
    currentThresholdValue = currentStruct.Rthr;
    centeredCurrentMatrix = shift_feature_matrix(currentFeatureMatrix, centeredFeatureMatrix);
    projectedMatrix = project_feature_matrix(centeredCurrentMatrix, principalDirections);
    projectedTwoColumn = extract_first_two_columns(projectedMatrix);
    isCriticalVector = determine_critical_indicator(currentTargetVector, currentThresholdValue);
    isNonCriticalVector = ~isCriticalVector;
    hold on;
    scatter(projectedTwoColumn(isNonCriticalVector, 1), ...
        projectedTwoColumn(isNonCriticalVector, 2), 12, colorLightGrayValue, ...
        'filled', 'MarkerFaceAlpha', 0.6);
    scatter(projectedTwoColumn(isCriticalVector, 1), ...
        projectedTwoColumn(isCriticalVector, 2), 25, colorPalette{datasetIndex}, ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.3);
    xlabel('PC1', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
    ylabel('PC2', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
    subplotTitle = sprintf('%s (PCA)', datasetNamesList{datasetIndex});
    title(subplotTitle, ...
        'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
    set(gca, 'FontSize', tickFontSizeValue, 'FontAngle', 'normal', 'XTickLabelRotation', 0);
    grid on; box on; xtickangle(0); hold off;
end

save_output_figure(outputDirectoryPath, 'fig_data_overview');

figureHandle = figure('Position', figurePositionValue, 'Color', 'w');
featureDimension = loadedDatasetContainer{1}.D;
for dimensionIndex = 1:featureDimension
    subplot(2, 3, dimensionIndex);
    hold on;
    for datasetIndex = 1:numberOfDatasets
        currentStruct = loadedDatasetContainer{datasetIndex};
        currentFeatureMatrix = currentStruct.X;
        currentFeatureColumn = extract_column_vector(currentFeatureMatrix, dimensionIndex);
        currentDisplayName = datasetNamesList{datasetIndex};
        currentColorValue = colorPalette{datasetIndex};
        histogram(currentFeatureColumn, 20, 'Normalization', 'pdf', ...
            'FaceColor', currentColorValue, 'FaceAlpha', 0.4, ...
            'DisplayName', currentDisplayName);
    end
    dimensionLabelText = sprintf('Dim %d', dimensionIndex);
    xlabel(dimensionLabelText, 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
    ylabel('Density', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
    subplotTitleText = sprintf('Marginal distribution: dim %d', dimensionIndex);
    title(subplotTitleText, ...
        'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
    set(gca, 'FontSize', tickFontSizeValue, 'FontAngle', 'normal', 'XTickLabelRotation', 0);
    if dimensionIndex == 1
        legend('Location', 'best', 'FontSize', legendFontSizeValue);
    end
    grid on; box on; xtickangle(0); hold off;
end

save_output_figure(outputDirectoryPath, 'fig_data_marginal');

end


function resultVisualizationRoutine(resultContainerStructure, datasetNamesList, ...
    colorPalette, colorLightGrayValue, figurePositionValue, ...
    titleFontSizeValue, labelFontSizeValue, tickFontSizeValue, ...
    legendFontSizeValue, outputDirectoryPath)

numberOfDatasets = numel(resultContainerStructure);

dataNameCellArray = cell(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    dataNameCellArray{datasetIndex} = resultContainerStructure(datasetIndex).datasetLabel;
end

figureHandle = figure('Position', figurePositionValue, 'Color', 'w');

subplot(2, 3, 1);
rmseValues = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    rmseValues(datasetIndex) = resultContainerStructure(datasetIndex).criticalRMSEValue;
end
barHandle = bar(rmseValues, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
for datasetIndex = 1:numberOfDatasets
    barHandle.CData(datasetIndex, :) = colorPalette{datasetIndex};
end
set(gca, 'XTickLabel', dataNameCellArray, 'FontSize', tickFontSizeValue, ...
    'FontAngle', 'normal', 'XTickLabelRotation', 0);
ylabel('RMSE', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Critical-region RMSE', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
maxRmseValue = compute_scalar_maximum(rmseValues);
for datasetIndex = 1:numberOfDatasets
    offsetValue = maxRmseValue * 0.03;
    labelYPosition = rmseValues(datasetIndex) + offsetValue;
    labelText = sprintf('%.4f', rmseValues(datasetIndex));
    text(datasetIndex, labelYPosition, labelText, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end

subplot(2, 3, 2);
maeValues = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    maeValues(datasetIndex) = resultContainerStructure(datasetIndex).criticalMAEValue;
end
barHandle = bar(maeValues, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
for datasetIndex = 1:numberOfDatasets
    barHandle.CData(datasetIndex, :) = colorPalette{datasetIndex};
end
set(gca, 'XTickLabel', dataNameCellArray, 'FontSize', tickFontSizeValue, ...
    'FontAngle', 'normal', 'XTickLabelRotation', 0);
ylabel('MAE', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Critical-region MAE', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
maxMaeValue = compute_scalar_maximum(maeValues);
for datasetIndex = 1:numberOfDatasets
    offsetValue = maxMaeValue * 0.03;
    labelYPosition = maeValues(datasetIndex) + offsetValue;
    labelText = sprintf('%.4f', maeValues(datasetIndex));
    text(datasetIndex, labelYPosition, labelText, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end

subplot(2, 3, 3);
f1Values = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    f1Values(datasetIndex) = resultContainerStructure(datasetIndex).f1ScoreValue;
end
barHandle = bar(f1Values, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
for datasetIndex = 1:numberOfDatasets
    barHandle.CData(datasetIndex, :) = colorPalette{datasetIndex};
end
set(gca, 'XTickLabel', dataNameCellArray, 'FontSize', tickFontSizeValue, ...
    'FontAngle', 'normal', 'XTickLabelRotation', 0);
ylabel('F1 score', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Critical-region F1', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
maxF1Value = compute_scalar_maximum(f1Values);
for datasetIndex = 1:numberOfDatasets
    offsetValue = maxF1Value * 0.03;
    labelYPosition = f1Values(datasetIndex) + offsetValue;
    labelText = sprintf('%.4f', f1Values(datasetIndex));
    text(datasetIndex, labelYPosition, labelText, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end

subplot(2, 3, 4);
hold on;
for datasetIndex = 1:numberOfDatasets
    recallValue = resultContainerStructure(datasetIndex).recallValue;
    precisionValue = resultContainerStructure(datasetIndex).precisionValue;
    scatter(recallValue, precisionValue, 200, colorPalette{datasetIndex}, ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
    labelXPosition = recallValue + 0.03;
    labelYPosition = precisionValue + 0.02;
    labelText = dataNameCellArray{datasetIndex};
    text(labelXPosition, labelYPosition, labelText, ...
        'FontWeight', 'bold', 'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end
xlabel('Recall', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
ylabel('Precision', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
xlim([0 1]); ylim([0 1.10]);
title('Precision-Recall', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
set(gca, 'FontSize', tickFontSizeValue, 'FontAngle', 'normal', 'XTickLabelRotation', 0);
grid on; box on; xtickangle(0); hold off;

subplot(2, 3, 5);
criticalCountValues = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    criticalCountValues(datasetIndex) = resultContainerStructure(datasetIndex).criticalCountValue;
end
barHandle = bar(criticalCountValues, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
for datasetIndex = 1:numberOfDatasets
    barHandle.CData(datasetIndex, :) = colorPalette{datasetIndex};
end
set(gca, 'XTickLabel', dataNameCellArray, 'FontSize', tickFontSizeValue, ...
    'FontAngle', 'normal', 'XTickLabelRotation', 0);
ylabel('Count', 'FontSize', labelFontSizeValue, 'FontAngle', 'normal');
title('Critical points in training set', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
maxCountValue = compute_scalar_maximum(criticalCountValues);
for datasetIndex = 1:numberOfDatasets
    offsetValue = maxCountValue * 0.03;
    labelYPosition = criticalCountValues(datasetIndex) + offsetValue;
    labelText = sprintf('%d', criticalCountValues(datasetIndex));
    text(datasetIndex, labelYPosition, labelText, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', tickFontSizeValue, 'FontAngle', 'normal');
end

subplot(2, 3, 6);
metricLabelsCellArray = {'1/RMSE', '1/MAE', 'F1', 'Precision', 'Recall', 'CritPts'};
minimumRmseValue = compute_scalar_minimum(rmseValues);
minimumMaeValue = compute_scalar_minimum(maeValues);
maximumF1Value = compute_scalar_maximum(f1Values);

precisionValues = zeros(1, numberOfDatasets);
recallValues = zeros(1, numberOfDatasets);
for datasetIndex = 1:numberOfDatasets
    precisionValues(datasetIndex) = resultContainerStructure(datasetIndex).precisionValue;
    recallValues(datasetIndex) = resultContainerStructure(datasetIndex).recallValue;
end
maximumPrecisionValue = compute_scalar_maximum(precisionValues);
maximumRecallValue = compute_scalar_maximum(recallValues);
maximumCountValue = compute_scalar_maximum(criticalCountValues);

scoreMatrix = zeros(numberOfDatasets, 6);
for datasetIndex = 1:numberOfDatasets
    scoreMatrix(datasetIndex, 1) = minimumRmseValue / rmseValues(datasetIndex);
    scoreMatrix(datasetIndex, 2) = minimumMaeValue / maeValues(datasetIndex);
    scoreMatrix(datasetIndex, 3) = f1Values(datasetIndex) / maximumF1Value;
    scoreMatrix(datasetIndex, 4) = precisionValues(datasetIndex) / maximumPrecisionValue;
    scoreMatrix(datasetIndex, 5) = recallValues(datasetIndex) / maximumRecallValue;
    scoreMatrix(datasetIndex, 6) = criticalCountValues(datasetIndex) / maximumCountValue;
end

numberOfColorLevels = 64;
colorMapMatrix = zeros(numberOfColorLevels, 3);
for colorLevelIndex = 1:numberOfColorLevels
    fractionValue = (colorLevelIndex - 1) / (numberOfColorLevels - 1);
    redComponentValue = 0.97 - fractionValue * (0.97 - 0.20);
    greenComponentValue = 0.97 - fractionValue * (0.97 - 0.45);
    blueComponentValue = 0.99 - fractionValue * (0.99 - 0.70);
    colorMapMatrix(colorLevelIndex, 1) = redComponentValue;
    colorMapMatrix(colorLevelIndex, 2) = greenComponentValue;
    colorMapMatrix(colorLevelIndex, 3) = blueComponentValue;
end

imagesc(scoreMatrix);
colormap(gca, colorMapMatrix);
colorBarHandle = colorbar;
colorBarHandle.Label.String = 'Normalized score';
colorBarHandle.Label.FontSize = tickFontSizeValue;
colorBarHandle.Label.FontAngle = 'normal';

set(gca, 'XTick', 1:6, 'XTickLabel', metricLabelsCellArray, ...
    'YTick', 1:numberOfDatasets, 'YTickLabel', dataNameCellArray, ...
    'FontSize', tickFontSizeValue, 'FontAngle', 'normal', ...
    'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
title('Normalized score matrix', ...
    'FontSize', titleFontSizeValue, 'FontWeight', 'bold', 'FontAngle', 'normal');
xtickangle(0); ytickangle(0);

for rowIndex = 1:numberOfDatasets
    for columnIndex = 1:6
        textColorValue = 'k';
        if scoreMatrix(rowIndex, columnIndex) > 0.65
            textColorValue = 'w';
        end
        cellText = sprintf('%.2f', scoreMatrix(rowIndex, columnIndex));
        text(columnIndex, rowIndex, cellText, ...
            'HorizontalAlignment', 'center', ...
            'FontSize', tickFontSizeValue, 'FontAngle', 'normal', ...
            'Color', textColorValue, 'FontWeight', 'bold');
    end
end

save_output_figure(outputDirectoryPath, 'fig_results_overview');

end


function forest = build_weighted_random_forest(featureMatrix, targetVector, ...
    thresholdValue, configurationStructure)

numberOfSamples = size(featureMatrix, 1);
weightVector = zeros(numberOfSamples, 1);
for sampleIndex = 1:numberOfSamples
    currentTargetValue = targetVector(sampleIndex);
    if currentTargetValue <= thresholdValue
        weightVector(sampleIndex) = configurationStructure.criticalWeightFactorValue;
    else
        weightVector(sampleIndex) = 1;
    end
end
weightSum = compute_scalar_sum(weightVector);
weightVector = weightVector / weightSum;

forestStructure.treeCount = configurationStructure.treeCountValue;
forestStructure.trees = cell(1, configurationStructure.treeCountValue);

for treeIndex = 1:configurationStructure.treeCountValue
    bootstrapIndices = generate_weighted_indices(weightVector, numberOfSamples);
    bootstrapFeatureMatrix = extract_rows_by_indices(featureMatrix, bootstrapIndices);
    bootstrapTargetVector = extract_values_by_indices(targetVector, bootstrapIndices);
    featureSubsetSize = compute_feature_subset_size(featureMatrix);
    forestStructure.trees{treeIndex} = fitrtree( ...
        bootstrapFeatureMatrix, bootstrapTargetVector, ...
        'MinLeafSize', configurationStructure.minimumLeafSizeValue, ...
        'NumVariablesToSample', featureSubsetSize, ...
        'MaxNumSplits', configurationStructure.maximumSplitCountValue, ...
        'Reproducible', true);
end

forest = forestStructure;
end


function predictions = evaluate_random_forest(forestStructure, featureMatrix)

numberOfRows = size(featureMatrix, 1);
accumulatedPredictions = zeros(forestStructure.treeCount, numberOfRows);

for treeIndex = 1:forestStructure.treeCount
    currentTree = forestStructure.trees{treeIndex};
    treePredictions = predict(currentTree, featureMatrix);
    for rowIndex = 1:numberOfRows
        accumulatedPredictions(treeIndex, rowIndex) = treePredictions(rowIndex);
    end
end

predictions = zeros(numberOfRows, 1);
for rowIndex = 1:numberOfRows
    columnValues = accumulatedPredictions(:, rowIndex);
    columnSum = compute_scalar_sum(columnValues);
    predictions(rowIndex) = columnSum / forestStructure.treeCount;
end
end


function indicatorVector = determine_critical_indicator(valueVector, thresholdValue)
numberOfElements = numel(valueVector);
indicatorVector = false(numberOfElements, 1);
for elementIndex = 1:numberOfElements
    if valueVector(elementIndex) <= thresholdValue
        indicatorVector(elementIndex) = true;
    end
end
end


function maskedValues = extract_masked_values(valueVector, maskVector)
numberOfElements = numel(valueVector);
maskedValues = zeros(numberOfElements, 1);
runningCounter = 0;
for elementIndex = 1:numberOfElements
    if maskVector(elementIndex)
        runningCounter = runningCounter + 1;
        maskedValues(runningCounter) = valueVector(elementIndex);
    end
end
maskedValues = maskedValues(1:runningCounter);
end


function countValue = compute_logical_and_count(vectorA, vectorB)
numberOfElements = numel(vectorA);
countValue = 0;
for elementIndex = 1:numberOfElements
    if vectorA(elementIndex) && vectorB(elementIndex)
        countValue = countValue + 1;
    end
end
end


function countValue = compute_logical_sum(inputVector)
numberOfElements = numel(inputVector);
countValue = 0;
for elementIndex = 1:numberOfElements
    if inputVector(elementIndex)
        countValue = countValue + 1;
    end
end
end


function meanValue = compute_scalar_mean(inputVector)
numberOfElements = numel(inputVector);
if numberOfElements == 0
    meanValue = NaN;
    return;
end
accumulatorValue = 0;
for elementIndex = 1:numberOfElements
    accumulatorValue = accumulatorValue + inputVector(elementIndex);
end
meanValue = accumulatorValue / numberOfElements;
end


function sumValue = compute_scalar_sum(inputVector)
numberOfElements = numel(inputVector);
accumulatorValue = 0;
for elementIndex = 1:numberOfElements
    accumulatorValue = accumulatorValue + inputVector(elementIndex);
end
sumValue = accumulatorValue;
end


function maxValue = compute_scalar_maximum(inputVector)
numberOfElements = numel(inputVector);
if numberOfElements == 0
    maxValue = NaN;
    return;
end
maxValue = inputVector(1);
for elementIndex = 2:numberOfElements
    if inputVector(elementIndex) > maxValue
        maxValue = inputVector(elementIndex);
    end
end
end


function minValue = compute_scalar_minimum(inputVector)
numberOfElements = numel(inputVector);
if numberOfElements == 0
    minValue = NaN;
    return;
end
minValue = inputVector(1);
for elementIndex = 2:numberOfElements
    if inputVector(elementIndex) < minValue
        minValue = inputVector(elementIndex);
    end
end
end


function outputIndices = generate_weighted_indices(weightVector, numberOfSamples)
normalizedWeights = weightVector / compute_scalar_sum(weightVector);
cumulativeWeights = zeros(numberOfSamples, 1);
runningSum = 0;
for elementIndex = 1:numberOfSamples
    runningSum = runningSum + normalizedWeights(elementIndex);
    cumulativeWeights(elementIndex) = runningSum;
end
outputIndices = zeros(numberOfSamples, 1);
for sampleIndex = 1:numberOfSamples
    uniformValue = rand();
    selectedIndex = numberOfSamples;
    for candidateIndex = 1:numberOfSamples
        if uniformValue <= cumulativeWeights(candidateIndex)
            selectedIndex = candidateIndex;
            break;
        end
    end
    if selectedIndex < 1
        selectedIndex = 1;
    end
    if selectedIndex > numberOfSamples
        selectedIndex = numberOfSamples;
    end
    outputIndices(sampleIndex) = selectedIndex;
end
end


function outputMatrix = extract_rows_by_indices(inputMatrix, indexVector)
numberOfRows = numel(indexVector);
numberOfColumns = size(inputMatrix, 2);
outputMatrix = zeros(numberOfRows, numberOfColumns);
for rowIndex = 1:numberOfRows
    sourceRowIndex = indexVector(rowIndex);
    for columnIndex = 1:numberOfColumns
        outputMatrix(rowIndex, columnIndex) = inputMatrix(sourceRowIndex, columnIndex);
    end
end
end


function outputVector = extract_values_by_indices(inputVector, indexVector)
numberOfRows = numel(indexVector);
outputVector = zeros(numberOfRows, 1);
for rowIndex = 1:numberOfRows
    sourceRowIndex = indexVector(rowIndex);
    outputVector(rowIndex) = inputVector(sourceRowIndex);
end
end


function featureSubsetSize = compute_feature_subset_size(featureMatrix)
numberOfColumns = size(featureMatrix, 2);
squareRootValue = sqrt(numberOfColumns);
flooredValue = floor(squareRootValue);
if flooredValue < 1
    featureSubsetSize = 1;
else
    featureSubsetSize = flooredValue;
end
end


function concatenatedMatrix = concatenate_feature_matrices(datasetContainer)
numberOfDatasets = numel(datasetContainer);
firstMatrix = datasetContainer{1}.X;
totalRows = 0;
for datasetIndex = 1:numberOfDatasets
    currentMatrix = datasetContainer{datasetIndex}.X;
    totalRows = totalRows + size(currentMatrix, 1);
end
numberOfColumns = size(firstMatrix, 2);
concatenatedMatrix = zeros(totalRows, numberOfColumns);
currentRowIndex = 0;
for datasetIndex = 1:numberOfDatasets
    currentMatrix = datasetContainer{datasetIndex}.X;
    currentRowCount = size(currentMatrix, 1);
    for rowIndex = 1:currentRowCount
        currentRowIndex = currentRowIndex + 1;
        for columnIndex = 1:numberOfColumns
            concatenatedMatrix(currentRowIndex, columnIndex) = ...
                currentMatrix(rowIndex, columnIndex);
        end
    end
end
end


function centeredMatrix = center_feature_matrix(inputMatrix)
numberOfRows = size(inputMatrix, 1);
numberOfColumns = size(inputMatrix, 2);
columnMeanVector = zeros(1, numberOfColumns);
for columnIndex = 1:numberOfColumns
    columnSum = 0;
    for rowIndex = 1:numberOfRows
        columnSum = columnSum + inputMatrix(rowIndex, columnIndex);
    end
    columnMeanVector(columnIndex) = columnSum / numberOfRows;
end
centeredMatrix = zeros(numberOfRows, numberOfColumns);
for rowIndex = 1:numberOfRows
    for columnIndex = 1:numberOfColumns
        centeredMatrix(rowIndex, columnIndex) = ...
            inputMatrix(rowIndex, columnIndex) - columnMeanVector(columnIndex);
    end
end
end


function shiftedMatrix = shift_feature_matrix(inputMatrix, referenceMatrix)
numberOfColumns = size(referenceMatrix, 2);
numberOfRows = size(referenceMatrix, 1);
columnMeanVector = zeros(1, numberOfColumns);
for columnIndex = 1:numberOfColumns
    columnSum = 0;
    for rowIndex = 1:numberOfRows
        columnSum = columnSum + referenceMatrix(rowIndex, columnIndex);
    end
    columnMeanVector(columnIndex) = columnSum / numberOfRows;
end
numberOfInputRows = size(inputMatrix, 1);
shiftedMatrix = zeros(numberOfInputRows, numberOfColumns);
for rowIndex = 1:numberOfInputRows
    for columnIndex = 1:numberOfColumns
        shiftedMatrix(rowIndex, columnIndex) = ...
            inputMatrix(rowIndex, columnIndex) - columnMeanVector(columnIndex);
    end
end
end


function principalDirections = extract_principal_directions(centeredMatrix)
[~, ~, rightSingularVectors] = svd(centeredMatrix, 'econ');
principalDirections = rightSingularVectors;
end


function projectedMatrix = project_feature_matrix(inputMatrix, directionMatrix)
numberOfRows = size(inputMatrix, 1);
numberOfColumns = size(directionMatrix, 2);
numberOfComponents = size(inputMatrix, 2);
projectedMatrix = zeros(numberOfRows, numberOfColumns);
for rowIndex = 1:numberOfRows
    for columnIndex = 1:numberOfColumns
        accumulatorValue = 0;
        for componentIndex = 1:numberOfComponents
            accumulatorValue = accumulatorValue + ...
                inputMatrix(rowIndex, componentIndex) * ...
                directionMatrix(componentIndex, columnIndex);
        end
        projectedMatrix(rowIndex, columnIndex) = accumulatorValue;
    end
end
end


function outputMatrix = extract_first_two_columns(inputMatrix)
numberOfRows = size(inputMatrix, 1);
numberOfAvailableColumns = size(inputMatrix, 2);
if numberOfAvailableColumns < 2
    outputMatrix = zeros(numberOfRows, 2);
    outputMatrix(:, 1) = inputMatrix(:, 1);
    return;
end
outputMatrix = zeros(numberOfRows, 2);
for rowIndex = 1:numberOfRows
    outputMatrix(rowIndex, 1) = inputMatrix(rowIndex, 1);
    outputMatrix(rowIndex, 2) = inputMatrix(rowIndex, 2);
end
end


function columnVector = extract_column_vector(inputMatrix, columnIndex)
numberOfRows = size(inputMatrix, 1);
columnVector = zeros(numberOfRows, 1);
for rowIndex = 1:numberOfRows
    columnVector(rowIndex) = inputMatrix(rowIndex, columnIndex);
end
end


function outputVector = duplicate_vector(inputVector)
numberOfElements = numel(inputVector);
outputVector = zeros(numberOfElements, 1);
for elementIndex = 1:numberOfElements
    outputVector(elementIndex) = inputVector(elementIndex);
end
end


function outputMatrix = duplicate_matrix(inputMatrix)
numberOfRows = size(inputMatrix, 1);
numberOfColumns = size(inputMatrix, 2);
outputMatrix = zeros(numberOfRows, numberOfColumns);
for rowIndex = 1:numberOfRows
    for columnIndex = 1:numberOfColumns
        outputMatrix(rowIndex, columnIndex) = inputMatrix(rowIndex, columnIndex);
    end
end
end


function outputVector = reverse_vector(inputVector)
numberOfElements = numel(inputVector);
outputVector = zeros(numberOfElements, 1);
for elementIndex = 1:numberOfElements
    outputVector(elementIndex) = inputVector(numberOfElements - elementIndex + 1);
end
end


function spanValue = compute_vector_span(inputVector)
maximumValue = compute_scalar_maximum(inputVector);
minimumValue = compute_scalar_minimum(inputVector);
spanValue = maximumValue - minimumValue;
end


function squaredNormValue = compute_vector_squared_norm(inputVector)
numberOfElements = numel(inputVector);
accumulatorValue = 0;
for elementIndex = 1:numberOfElements
    squaredElement = inputVector(elementIndex) .^ 2;
    accumulatorValue = accumulatorValue + squaredElement;
end
squaredNormValue = accumulatorValue;
end


function outputVector = clip_vector_to_bounds(inputVector, lowerBound, upperBound)
numberOfElements = numel(inputVector);
outputVector = zeros(numberOfElements, 1);
for elementIndex = 1:numberOfElements
    currentValue = inputVector(elementIndex);
    if currentValue < lowerBound
        outputVector(elementIndex) = lowerBound;
    elseif currentValue > upperBound
        outputVector(elementIndex) = upperBound;
    else
        outputVector(elementIndex) = currentValue;
    end
end
end


function divisionResult = perform_safe_division(numeratorValue, denominatorValue)
if denominatorValue == 0
    divisionResult = 0;
    return;
end
divisionResult = numeratorValue / denominatorValue;
end


function emptyCheckResult = check_container_empty(inputContainer)
emptyCheckResult = isempty(inputContainer);
end


function positiveCheckResult = verify_positive_value(inputValue)
positiveCheckResult = inputValue > 0;
end


function sizeMatchResult = verify_matching_size(matrixA, matrixB)
numberOfRowsA = size(matrixA, 1);
numberOfRowsB = size(matrixB, 1);
sizeMatchResult = (numberOfRowsA == numberOfRowsB);
end


function copiedCellArray = duplicate_cell_array(inputCellArray)
numberOfElements = numel(inputCellArray);
copiedCellArray = cell(1, numberOfElements);
for elementIndex = 1:numberOfElements
    copiedCellArray{elementIndex} = inputCellArray{elementIndex};
end
end


function outputVector = create_sequential_indices(numberOfElements)
outputVector = zeros(1, numberOfElements);
for elementIndex = 1:numberOfElements
    outputVector(elementIndex) = elementIndex;
end
end


function outputVector = reverse_sequential_indices(inputVector)
numberOfElements = numel(inputVector);
outputVector = zeros(1, numberOfElements);
for elementIndex = 1:numberOfElements
    outputVector(elementIndex) = inputVector(numberOfElements - elementIndex + 1);
end
end


function outputValue = extract_numeric_from_cell(inputCell, cellIndex)
contentValue = inputCell{cellIndex};
if isnumeric(contentValue)
    outputValue = contentValue;
else
    outputValue = NaN;
end
end


function save_output_figure(outputDirectoryPath, baseNameString)
set(gcf, 'Color', 'w');
fileNameWithExtension = strcat(baseNameString, '.png');
fullOutputPath = fullfile(outputDirectoryPath, fileNameWithExtension);
print(gcf, fullOutputPath, '-dpng', '-r300');
end