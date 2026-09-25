function surrogate_strategy_benchmark()

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultFigurePaperPositionMode', 'auto');
set(0, 'DefaultFigureInvertHardcopy', 'off');

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

randomSeedValue = 42;
rng(randomSeedValue);

problemDimension = 15;
lowerBoundValue = -3;
upperBoundValue = 3;

initialPoolSize = 300;
batchIncrementSize = 20;
totalBatchCount = 50;
updateTriggerLevel = 100;
evaluationSetSize = 3000;
noiseStandardDeviation = 0.3;

numberOfTrees = 100;
baselineMinimumLeafSize = 1;
hiddenLayerConfiguration = [128, 64, 64, 32, 16];
trainingEpochBudget = 60;

runtimeOptions = struct( ...
    'defaultSeedOffset', 7, ...
    'validationSplitRatio', 0.30, ...
    'bootstrapSampleFraction', 1.0, ...
    'minimumValidationSize', 20, ...
    'activationFunctionFlag', 1, ...
    'weightInitializationScale', 2.0, ...
    'miniBatchSizeValue', 32, ...
    'learningRateValue', 0.003, ...
    'momentumCoefficientOne', 0.9, ...
    'momentumCoefficientTwo', 0.999, ...
    'numericalStabilityEpsilon', 1e-8, ...
    'kernelBandwidthScale', 0.5, ...
    'distanceThresholdValue', 1.5, ...
    'similarityCutoff', 0.75, ...
    'clusterTargetCount', 8, ...
    'outlierFraction', 0.05, ...
    'winsorizationLevel', 0.01, ...
    'varianceFilterThreshold', 1e-8, ...
    'correlationFilterCutoff', 0.95, ...
    'featureSelectionThreshold', 0.01, ...
    'driftDetectionBins', 6, ...
    'weightedSampleFactor', 6, ...
    'residualGateThreshold', 0.50, ...
    'leafExpansionLevel', 5, ...
    'retrainFractionValue', 0.30, ...
    'suspensionCooldownBatches', 3, ...
    'predictionBatchSize', 256, ...
    'cacheFlushInterval', 1000, ...
    'gradientClipLimit', 5.0, ...
    'parameterDecayFactor', 0.9999, ...
    'warmUpBatchCount', 5, ...
    'cooldownBatchCount', 3, ...
    'memoryPoolCapacity', 4096, ...
    'bufferReallocationStep', 512, ...
    'matrixFillValue', 0, ...
    'indexBaseOffset', 1, ...
    'loopingCounterInit', 0, ...
    'treeSplitCapValue', 100, ...
    'residualCooldownFactor', 1.25, ...
    'driftSmoothingWeight', 0.85);

is_valid_runtime_options(runtimeOptions);

totalStreamLength = initialPoolSize + batchIncrementSize * totalBatchCount;
sourceFeatureMatrix = zeros(totalStreamLength, problemDimension);
sourceTargetVector = zeros(totalStreamLength, 1);

regimeBreakpointFractions = [0.20, 0.40, 0.60, 0.80];
regimeBreakpointIndices = round(totalStreamLength * regimeBreakpointFractions);
regimeCenterMatrix = [ 0,   0,   0;
                       2.5, -2,  1;
                      -2.5, 2.5, -1;
                       2,   2,   2;
                      -2,  -2,   2];

for sampleIndex = 1:totalStreamLength
    currentRegime = 1;
    if sampleIndex > regimeBreakpointIndices(1)
        currentRegime = 2;
    end
    if sampleIndex > regimeBreakpointIndices(2)
        currentRegime = 3;
    end
    if sampleIndex > regimeBreakpointIndices(3)
        currentRegime = 4;
    end
    if sampleIndex > regimeBreakpointIndices(4)
        currentRegime = 5;
    end
    currentCenterVector = regimeCenterMatrix(currentRegime, :);
    currentCenterLength = numel(currentCenterVector);
    if currentCenterLength < problemDimension
        paddingLength = problemDimension - currentCenterLength;
        paddingVector = zeros(1, paddingLength);
        currentCenterVector = [currentCenterVector, paddingVector];
    end
    uniformRandomRow = rand(1, problemDimension);
    scaledRandomRow = uniformRandomRow * (upperBoundValue - lowerBoundValue);
    shiftedRandomRow = scaledRandomRow + lowerBoundValue;
    rawFeatureRow = currentCenterVector + shiftedRandomRow;
    clippedFeatureRow = max(lowerBoundValue, min(upperBoundValue, rawFeatureRow));
    sourceFeatureMatrix(sampleIndex, :) = clippedFeatureRow;
    latentValue = latent_function(clippedFeatureRow, problemDimension);
    noiseContribution = noiseStandardDeviation * randn();
    sourceTargetVector(sampleIndex) = latentValue + noiseContribution;
end

evaluationFeatureMatrix = lb_random_matrix(evaluationSetSize, problemDimension, ...
    lowerBoundValue, upperBoundValue);
evaluationTargetVector = latent_function(evaluationFeatureMatrix, problemDimension);

modelDisplayNames = {'Full-batch RF', 'Incremental Strat RF', 'Neural Network'};
modelColorPalette = { [0.00 0.45 0.74], [0.90 0.10 0.10], [0.47 0.67 0.19] };
modelCounter = 3;

mseProgressionMatrix = NaN(modelCounter, totalBatchCount + 1);
trainingTimeMatrix = NaN(modelCounter, totalBatchCount + 1);
predictionTimeMatrix = NaN(modelCounter, totalBatchCount + 1);
accumulatedTimeMatrix = NaN(modelCounter, totalBatchCount + 1);
relativeEfficiencyMatrix = NaN(modelCounter, totalBatchCount + 1);
driftHistoryVector = NaN(1, totalBatchCount + 1);
leafSizeHistoryVector = NaN(1, totalBatchCount + 1);
rmseHistoryVector = NaN(1, totalBatchCount + 1);
suspensionHistoryVector = NaN(1, totalBatchCount + 1);

workingFeaturePool = sourceFeatureMatrix(1:initialPoolSize, :);
workingTargetPool = sourceTargetVector(1:initialPoolSize);

clockHandle = tic;
fullBatchForestModel = assemble_full_forest(workingFeaturePool, workingTargetPool, ...
    numberOfTrees, baselineMinimumLeafSize, 100);
trainingTimeMatrix(1, 1) = toc(clockHandle);

clockHandle = tic;
predictedValues = infer_full_forest(fullBatchForestModel, evaluationFeatureMatrix);
predictionTimeMatrix(1, 1) = toc(clockHandle);
mseProgressionMatrix(1, 1) = mean_squared_deviation( ...
    evaluationTargetVector, predictedValues);

incrementalForestModel = establish_incremental_model(numberOfTrees, problemDimension, ...
    0.30, updateTriggerLevel);
clockHandle = tic;
incrementalForestModel = initialize_incremental_pool(incrementalForestModel, ...
    workingFeaturePool, workingTargetPool);
trainingTimeMatrix(2, 1) = toc(clockHandle);

clockHandle = tic;
predictedValues = infer_incremental_pool(incrementalForestModel, evaluationFeatureMatrix);
predictionTimeMatrix(2, 1) = toc(clockHandle);
mseProgressionMatrix(2, 1) = mean_squared_deviation( ...
    evaluationTargetVector, predictedValues);

clockHandle = tic;
neuralNetworkModel = mlp_initialize(problemDimension, hiddenLayerConfiguration);
neuralNetworkModel = mlp_set_normalization(neuralNetworkModel, ...
    workingFeaturePool, workingTargetPool);
neuralNetworkModel = mlp_train_adam(neuralNetworkModel, ...
    workingFeaturePool, workingTargetPool, trainingEpochBudget);
trainingTimeMatrix(3, 1) = toc(clockHandle);

clockHandle = tic;
predictedValues = mlp_predict(neuralNetworkModel, evaluationFeatureMatrix);
predictionTimeMatrix(3, 1) = toc(clockHandle);
mseProgressionMatrix(3, 1) = mean_squared_deviation( ...
    evaluationTargetVector, predictedValues);

accumulatedTimeMatrix(:, 1) = trainingTimeMatrix(:, 1) + predictionTimeMatrix(:, 1);

for currentBatchIndex = 1:totalBatchCount
    streamStartIndex = initialPoolSize + (currentBatchIndex - 1) * batchIncrementSize + 1;
    streamEndIndex = streamStartIndex + batchIncrementSize - 1;

    incomingFeatureChunk = sourceFeatureMatrix(streamStartIndex:streamEndIndex, :);
    incomingTargetChunk = sourceTargetVector(streamStartIndex:streamEndIndex);

    workingFeaturePool = append_rows_to_matrix(workingFeaturePool, incomingFeatureChunk);
    workingTargetPool = append_values_to_vector(workingTargetPool, incomingTargetChunk);

    clockHandle = tic;
    fullBatchForestModel = assemble_full_forest(workingFeaturePool, workingTargetPool, ...
        numberOfTrees, baselineMinimumLeafSize, 100);
    trainingTimeMatrix(1, currentBatchIndex + 1) = toc(clockHandle);

    clockHandle = tic;
    predictedValues = infer_full_forest(fullBatchForestModel, evaluationFeatureMatrix);
    predictionTimeMatrix(1, currentBatchIndex + 1) = toc(clockHandle);
    mseProgressionMatrix(1, currentBatchIndex + 1) = mean_squared_deviation( ...
        evaluationTargetVector, predictedValues);

    clockHandle = tic;
    [incrementalForestModel, updateReport] = refresh_incremental_pool( ...
        incrementalForestModel, incomingFeatureChunk, incomingTargetChunk);
    trainingTimeMatrix(2, currentBatchIndex + 1) = toc(clockHandle);

    clockHandle = tic;
    predictedValues = infer_incremental_pool(incrementalForestModel, evaluationFeatureMatrix);
    predictionTimeMatrix(2, currentBatchIndex + 1) = toc(clockHandle);
    mseProgressionMatrix(2, currentBatchIndex + 1) = mean_squared_deviation( ...
        evaluationTargetVector, predictedValues);

    if updateReport.fired
        driftHistoryVector(currentBatchIndex + 1) = updateReport.div;
        leafSizeHistoryVector(currentBatchIndex + 1) = updateReport.leaf;
        rmseHistoryVector(currentBatchIndex + 1) = updateReport.res;
        suspensionHistoryVector(currentBatchIndex + 1) = double(updateReport.pause);
    elseif currentBatchIndex > 1
        driftHistoryVector(currentBatchIndex + 1) = driftHistoryVector(currentBatchIndex);
        leafSizeHistoryVector(currentBatchIndex + 1) = leafSizeHistoryVector(currentBatchIndex);
        rmseHistoryVector(currentBatchIndex + 1) = rmseHistoryVector(currentBatchIndex);
        suspensionHistoryVector(currentBatchIndex + 1) = suspensionHistoryVector(currentBatchIndex);
    end

    clockHandle = tic;
    neuralNetworkModel = mlp_initialize(problemDimension, hiddenLayerConfiguration);
    neuralNetworkModel = mlp_set_normalization(neuralNetworkModel, ...
        workingFeaturePool, workingTargetPool);
    neuralNetworkModel = mlp_train_adam(neuralNetworkModel, ...
        workingFeaturePool, workingTargetPool, trainingEpochBudget);
    trainingTimeMatrix(3, currentBatchIndex + 1) = toc(clockHandle);

    clockHandle = tic;
    predictedValues = mlp_predict(neuralNetworkModel, evaluationFeatureMatrix);
    predictionTimeMatrix(3, currentBatchIndex + 1) = toc(clockHandle);
    mseProgressionMatrix(3, currentBatchIndex + 1) = mean_squared_deviation( ...
        evaluationTargetVector, predictedValues);

    accumulatedTimeMatrix(:, currentBatchIndex + 1) = ...
        accumulatedTimeMatrix(:, currentBatchIndex) + ...
        trainingTimeMatrix(:, currentBatchIndex + 1) + ...
        predictionTimeMatrix(:, currentBatchIndex + 1);
end

for currentBatchIndex = 1:totalBatchCount + 1
    for currentModelIndex = 1:modelCounter
        baselineMSE = mseProgressionMatrix(1, currentBatchIndex);
        currentMSE = mseProgressionMatrix(currentModelIndex, currentBatchIndex);
        baselineTime = accumulatedTimeMatrix(1, currentBatchIndex);
        currentTime = accumulatedTimeMatrix(currentModelIndex, currentBatchIndex);
        if baselineMSE > 0
            if currentMSE > 0
                if baselineTime > 0
                    if currentTime > 0
                        mseRatio = baselineMSE / currentMSE;
                        timeRatio = currentTime / baselineTime;
                        relativeEfficiencyMatrix(currentModelIndex, currentBatchIndex) = ...
                            mseRatio / timeRatio;
                    end
                end
            end
        end
    end
end

finalFullForestPredictions = infer_full_forest(fullBatchForestModel, evaluationFeatureMatrix);
finalIncrementalPredictions = infer_incremental_pool(incrementalForestModel, evaluationFeatureMatrix);
finalNeuralPredictions = mlp_predict(neuralNetworkModel, evaluationFeatureMatrix);

finalFullResiduals = evaluationTargetVector - finalFullForestPredictions;
finalIncrementalResiduals = evaluationTargetVector - finalIncrementalPredictions;
finalNeuralResiduals = evaluationTargetVector - finalNeuralPredictions;

horizontalAxis = 0:totalBatchCount;

figureHandle = figure('Position', [100, 100, 1500, 500], 'Color', 'w');

subplot(2, 3, 1);
hold on;
for mm = 1:modelCounter
    if mm == 2
        lineWidthValue = 3;
    else
        lineWidthValue = 2;
    end
    plot(horizontalAxis, mseProgressionMatrix(mm, :), 'o-', ...
        'Color', modelColorPalette{mm}, ...
        'LineWidth', lineWidthValue, ...
        'MarkerFaceColor', modelColorPalette{mm}, ...
        'MarkerSize', 4, ...
        'DisplayName', modelDisplayNames{mm});
end
set(gca, 'YScale', 'log');
xlabel('Update batch');
ylabel('Test MSE');
legend('Location', 'northeast');
grid on;
box on;
hold off;

subplot(2, 3, 2);
hold on;
for mm = 1:modelCounter
    plot(horizontalAxis, accumulatedTimeMatrix(mm, :), 'o-', ...
        'Color', modelColorPalette{mm}, ...
        'LineWidth', 2, ...
        'MarkerFaceColor', modelColorPalette{mm}, ...
        'MarkerSize', 4, ...
        'DisplayName', modelDisplayNames{mm});
end
xlabel('Update batch');
ylabel('Cumulative time (s)');
legend('Location', 'northwest');
grid on;
box on;
hold off;

subplot(2, 3, 3);
hold on;
for mm = 1:modelCounter
    plot(horizontalAxis, relativeEfficiencyMatrix(mm, :), 'o-', ...
        'Color', modelColorPalette{mm}, ...
        'LineWidth', 2, ...
        'MarkerFaceColor', modelColorPalette{mm}, ...
        'MarkerSize', 4, ...
        'DisplayName', modelDisplayNames{mm});
end
yline(1.0, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Baseline');
xlabel('Update batch');
ylabel('Relative efficiency');
legend('Location', 'northwest');
grid on;
box on;
hold off;

subplot(2, 3, 4);
histogram(finalFullResiduals, 40, 'Normalization', 'pdf', ...
    'FaceColor', modelColorPalette{1}, 'FaceAlpha', 0.55);
xlabel('Prediction error');
ylabel('Probability density');
title(modelDisplayNames{1});
grid on;
box on;

subplot(2, 3, 5);
histogram(finalIncrementalResiduals, 40, 'Normalization', 'pdf', ...
    'FaceColor', modelColorPalette{2}, 'FaceAlpha', 0.55);
xlabel('Prediction error');
ylabel('Probability density');
title(modelDisplayNames{2});
grid on;
box on;

subplot(2, 3, 6);
histogram(finalNeuralResiduals, 40, 'Normalization', 'pdf', ...
    'FaceColor', modelColorPalette{3}, 'FaceAlpha', 0.55);
xlabel('Prediction error');
ylabel('Probability density');
title(modelDisplayNames{3});
grid on;
box on;

set(figureHandle, 'Color', 'w');
set(figureHandle, 'InvertHardcopy', 'off');
set(figureHandle, 'PaperPositionMode', 'auto');
outputFileName = 'benchmark_summary.png';
outputFilePath = fullfile(outputDirectoryPath, outputFileName);
print(figureHandle, outputFilePath, '-dpng', '-r300');

end


function flag = is_valid_runtime_options(options)
flag = true;
flag = flag && (options.defaultSeedOffset >= 0);
flag = flag && (options.validationSplitRatio > 0) && (options.validationSplitRatio < 1);
flag = flag && (options.bootstrapSampleFraction > 0);
flag = flag && (options.minimumValidationSize >= 1);
flag = flag && (options.activationFunctionFlag >= 0);
flag = flag && (options.weightInitializationScale > 0);
flag = flag && (options.miniBatchSizeValue >= 1);
flag = flag && (options.learningRateValue > 0);
flag = flag && (options.momentumCoefficientOne > 0) && (options.momentumCoefficientOne < 1);
flag = flag && (options.momentumCoefficientTwo > 0) && (options.momentumCoefficientTwo < 1);
flag = flag && (options.numericalStabilityEpsilon > 0);
flag = flag && (options.kernelBandwidthScale > 0);
flag = flag && (options.distanceThresholdValue > 0);
flag = flag && (options.similarityCutoff > 0) && (options.similarityCutoff < 1);
flag = flag && (options.clusterTargetCount >= 1);
flag = flag && (options.outlierFraction >= 0) && (options.outlierFraction < 1);
flag = flag && (options.winsorizationLevel >= 0) && (options.winsorizationLevel < 1);
flag = flag && (options.varianceFilterThreshold > 0);
flag = flag && (options.correlationFilterCutoff > 0) && (options.correlationFilterCutoff <= 1);
flag = flag && (options.featureSelectionThreshold >= 0);
flag = flag && (options.driftDetectionBins >= 1);
flag = flag && (options.weightedSampleFactor >= 1);
flag = flag && (options.residualGateThreshold > 0);
flag = flag && (options.leafExpansionLevel >= 1);
flag = flag && (options.retrainFractionValue > 0) && (options.retrainFractionValue <= 1);
flag = flag && (options.suspensionCooldownBatches >= 0);
flag = flag && (options.predictionBatchSize >= 1);
flag = flag && (options.cacheFlushInterval >= 1);
flag = flag && (options.gradientClipLimit > 0);
flag = flag && (options.parameterDecayFactor > 0) && (options.parameterDecayFactor <= 1);
flag = flag && (options.warmUpBatchCount >= 0);
flag = flag && (options.cooldownBatchCount >= 0);
flag = flag && (options.memoryPoolCapacity >= 1);
flag = flag && (options.bufferReallocationStep >= 1);
flag = flag && (options.matrixFillValue >= 0);
flag = flag && (options.indexBaseOffset >= 0);
flag = flag && (options.loopingCounterInit >= 0);
flag = flag && (options.treeSplitCapValue >= 1);
flag = flag && (options.residualCooldownFactor >= 1);
flag = flag && (options.driftSmoothingWeight > 0) && (options.driftSmoothingWeight <= 1);
if ~flag
    error('runtime option validation failed');
end
end


function val = latent_function(inputBlock, featureDim)
nRows = size(inputBlock, 1);
lowerEnd = min(5, featureDim);
upperStart = min(5, featureDim) + 1;
upperEnd = min(10, featureDim);
lowFrequencyComponent = mean(sin(2 * inputBlock(:, 1:lowerEnd)), 2);
highFrequencyComponent = mean(cos(4 * inputBlock(:, upperStart:upperEnd)), 2);
if featureDim >= 3
    interactionTerm = inputBlock(:, 1) .* inputBlock(:, 2) .* inputBlock(:, 3) / 9;
else
    interactionTerm = inputBlock(:, 1) .* inputBlock(:, 2) / 9;
end
quadraticTerm = 0.5 * mean(inputBlock .^ 2, 2) / 3;
if featureDim >= 4
    crossTermA = sin(inputBlock(:, 1) .* inputBlock(:, 4));
    crossTermB = cos(inputBlock(:, 2) .* inputBlock(:, 3));
    crossTerm = (crossTermA + crossTermB) / 2;
else
    crossTerm = zeros(nRows, 1);
end
linearTermA = 0.3 * inputBlock(:, 1) / 3;
linearTermB = 0.2 * inputBlock(:, 2) / 3;
linearTerm = linearTermA + linearTermB;
val = lowFrequencyComponent + 0.7 * highFrequencyComponent + ...
      0.5 * interactionTerm + 0.4 * quadraticTerm + ...
      0.6 * crossTerm + linearTerm;
end


function val = mean_squared_deviation(referenceValues, predictedValues)
differenceVector = referenceValues - predictedValues;
squaredDifferenceVector = differenceVector .^ 2;
squaredDifferenceMean = mean(squaredDifferenceVector);
val = squaredDifferenceMean;
end


function outMatrix = lb_random_matrix(rowCount, columnCount, lowBound, highBound)
outMatrix = zeros(rowCount, columnCount);
span = highBound - lowBound;
for columnIndex = 1:columnCount
    for rowIndex = 1:rowCount
        uniformValue = rand();
        scaledValue = uniformValue * span;
        shiftedValue = scaledValue + lowBound;
        outMatrix(rowIndex, columnIndex) = shiftedValue;
    end
end
end


function outMatrix = append_rows_to_matrix(existingMatrix, additionalRows)
existingRowCount = size(existingMatrix, 1);
additionalRowCount = size(additionalRows, 1);
existingColumnCount = size(existingMatrix, 2);
additionalColumnCount = size(additionalRows, 2);
if existingRowCount == 0
    outMatrix = additionalRows;
    return;
end
if additionalRowCount == 0
    outMatrix = existingMatrix;
    return;
end
if existingColumnCount ~= additionalColumnCount
    error('column count mismatch in append_rows_to_matrix');
end
combinedRowCount = existingRowCount + additionalRowCount;
outMatrix = zeros(combinedRowCount, existingColumnCount);
for rowIndex = 1:existingRowCount
    for columnIndex = 1:existingColumnCount
        outMatrix(rowIndex, columnIndex) = existingMatrix(rowIndex, columnIndex);
    end
end
for rowIndex = 1:additionalRowCount
    for columnIndex = 1:existingColumnCount
        outMatrix(existingRowCount + rowIndex, columnIndex) = ...
            additionalRows(rowIndex, columnIndex);
    end
end
end


function outVector = append_values_to_vector(existingVector, additionalValues)
existingLength = numel(existingVector);
additionalLength = numel(additionalValues);
if existingLength == 0
    outVector = additionalValues;
    return;
end
if additionalLength == 0
    outVector = existingVector;
    return;
end
combinedLength = existingLength + additionalLength;
outVector = zeros(combinedLength, 1);
for index = 1:existingLength
    outVector(index) = existingVector(index);
end
for index = 1:additionalLength
    outVector(existingLength + index) = additionalValues(index);
end
end


function tree = grow_decision_tree(X, y, nSamples, minLeafSize, maxSplitCount)
bootstrapIndices = randi(nSamples, nSamples, 1);
bootstrapFeatures = zeros(nSamples, size(X, 2));
bootstrapTargets = zeros(nSamples, 1);
for rowIndex = 1:nSamples
    sourceRowIndex = bootstrapIndices(rowIndex);
    bootstrapFeatures(rowIndex, :) = X(sourceRowIndex, :);
    bootstrapTargets(rowIndex) = y(sourceRowIndex);
end
featureSubsetSize = floor(sqrt(size(X, 2)));
if featureSubsetSize < 1
    featureSubsetSize = 1;
end
if isinf(maxSplitCount)
    tree = fitrtree(bootstrapFeatures, bootstrapTargets, ...
        'MinLeafSize', minLeafSize, ...
        'NumVariablesToSample', featureSubsetSize, ...
        'Reproducible', true);
else
    tree = fitrtree(bootstrapFeatures, bootstrapTargets, ...
        'MinLeafSize', minLeafSize, ...
        'MaxNumSplits', maxSplitCount, ...
        'NumVariablesToSample', featureSubsetSize, ...
        'Reproducible', true);
end
end


function tree = grow_weighted_decision_tree(X, y, nSamples, weightVector, minLeafSize)
normalizedWeights = weightVector(:) / sum(weightVector);
cumulativeWeights = cumsum(normalizedWeights);
randomDraws = rand(nSamples, 1);
chosenIndices = zeros(nSamples, 1);
for drawIndex = 1:nSamples
    drawValue = randomDraws(drawIndex);
    selectionIndex = nSamples;
    for candidateIndex = 1:nSamples
        if drawValue <= cumulativeWeights(candidateIndex)
            selectionIndex = candidateIndex;
            break;
        end
    end
    chosenIndices(drawIndex) = selectionIndex;
end
for index = 1:nSamples
    if chosenIndices(index) < 1
        chosenIndices(index) = 1;
    end
    if chosenIndices(index) > nSamples
        chosenIndices(index) = nSamples;
    end
end
sampledFeatures = zeros(nSamples, size(X, 2));
sampledTargets = zeros(nSamples, 1);
for rowIndex = 1:nSamples
    sourceRowIndex = chosenIndices(rowIndex);
    sampledFeatures(rowIndex, :) = X(sourceRowIndex, :);
    sampledTargets(rowIndex) = y(sourceRowIndex);
end
featureSubsetSize = floor(sqrt(size(X, 2)));
if featureSubsetSize < 1
    featureSubsetSize = 1;
end
tree = fitrtree(sampledFeatures, sampledTargets, ...
    'MinLeafSize', minLeafSize, ...
    'NumVariablesToSample', featureSubsetSize, ...
    'Reproducible', true);
end


function divValue = distribution_divergence(oldBlock, newBlock, nBins)
if isempty(oldBlock)
    divValue = 0;
    return;
end
if isempty(newBlock)
    divValue = 0;
    return;
end
featureCount = size(oldBlock, 2);
accumulatedDivergence = zeros(featureCount, 1);
for featureIndex = 1:featureCount
    oldColumn = oldBlock(:, featureIndex);
    newColumn = newBlock(:, featureIndex);
    combinedLower = min([oldColumn; newColumn]);
    combinedUpper = max([oldColumn; newColumn]);
    rangeValue = combinedUpper - combinedLower;
    if rangeValue < 1e-9
        continue;
    end
    binEdges = linspace(combinedLower, combinedUpper, nBins + 1);
    oldHistogram = histcounts(oldColumn, binEdges, 'Normalization', 'probability');
    newHistogram = histcounts(newColumn, binEdges, 'Normalization', 'probability');
    oldHistogram = oldHistogram + 1e-6;
    newHistogram = newHistogram + 1e-6;
    oldHistogram = oldHistogram / sum(oldHistogram);
    newHistogram = newHistogram / sum(newHistogram);
    forwardTerm = sum(oldHistogram .* log(oldHistogram ./ newHistogram));
    backwardTerm = sum(newHistogram .* log(newHistogram ./ oldHistogram));
    symmetricTerm = 0.5 * forwardTerm + 0.5 * backwardTerm;
    accumulatedDivergence(featureIndex) = symmetricTerm;
end
divValue = mean(accumulatedDivergence);
end


function forest = assemble_full_forest(X, y, treeCount, minLeafSize, maxSplitCount)
forest.treeCount = treeCount;
forest.trees = cell(1, treeCount);
nSamples = size(X, 1);
for treeIndex = 1:treeCount
    forest.trees{treeIndex} = grow_decision_tree(X, y, nSamples, ...
        minLeafSize, maxSplitCount);
end
end


function predictions = infer_full_forest(forest, X)
nRows = size(X, 1);
accumulatedPredictions = zeros(forest.treeCount, nRows);
for treeIndex = 1:forest.treeCount
    currentTree = forest.trees{treeIndex};
    treePredictions = predict(currentTree, X);
    accumulatedPredictions(treeIndex, :) = treePredictions';
end
predictions = mean(accumulatedPredictions, 1)';
end


function model = establish_incremental_model(treeCount, featureDim, retrainRatio, triggerLevel)
model.treeCount = treeCount;
retrainCount = round(treeCount * retrainRatio);
if retrainCount < 1
    retrainCount = 1;
end
model.retrainCount = retrainCount;
model.triggerLevel = triggerLevel;
model.featureDim = featureDim;
model.trees = cell(1, treeCount);
model.poolOldX = [];
model.poolOldY = [];
model.poolNewX = [];
model.poolNewY = [];
model.dklThreshold = 0.20;
model.newWeightFactor = 6;
model.residualThreshold = 0.50;
model.paused = false;
end


function model = initialize_incremental_pool(model, X, y)
model.poolOldX = X;
model.poolOldY = y;
nSamples = size(X, 1);
for treeIndex = 1:model.treeCount
    model.trees{treeIndex} = grow_decision_tree(X, y, nSamples, 1, inf);
end
end


function [model, report] = refresh_incremental_pool(model, Xchunk, ychunk)
report.fired = false;
report.div = NaN;
report.leaf = NaN;
report.res = NaN;
report.pause = false;

model.poolNewX = append_rows_to_matrix(model.poolNewX, Xchunk);
model.poolNewY = append_values_to_vector(model.poolNewY, ychunk);

newPoolSize = size(model.poolNewX, 1);
if newPoolSize < model.triggerLevel
    return;
end

nOld = size(model.poolOldX, 1);
nNew = size(model.poolNewX, 1);

divergenceValue = distribution_divergence(model.poolOldX, model.poolNewX, 6);

if divergenceValue > model.dklThreshold
    leafSetting = 5;
else
    leafSetting = 1;
end

Xall = append_rows_to_matrix(model.poolOldX, model.poolNewX);
yall = append_values_to_vector(model.poolOldY, model.poolNewY);
nAll = nOld + nNew;

weightVector = zeros(nAll, 1);
for index = 1:nOld
    weightVector(index) = 1;
end
for index = nOld + 1:nAll
    weightVector(index) = model.newWeightFactor;
end
weightVector = weightVector / sum(weightVector);

for retrainIndex = 1:model.retrainCount
    treeIndex = randi(model.treeCount);
    model.trees{treeIndex} = grow_weighted_decision_tree( ...
        Xall, yall, nAll, weightVector, leafSetting);
end

[residualValue, normalizedResidual] = assess_surrogate_accuracy(Xall, yall, model.treeCount);
model.paused = normalizedResidual > model.residualThreshold;

model.poolOldX = Xall;
model.poolOldY = yall;
model.poolNewX = [];
model.poolNewY = [];

report.fired = true;
report.div = divergenceValue;
report.leaf = leafSetting;
report.res = residualValue;
report.pause = model.paused;
end


function [residualValue, normalizedResidual] = assess_surrogate_accuracy(Xall, yall, treeCount)
nAll = size(Xall, 1);
if nAll < 20
    residualValue = NaN;
    normalizedResidual = NaN;
    return;
end
permutationVector = randperm(nAll);
nTraining = round(0.7 * nAll);
trainingIndices = permutationVector(1:nTraining);
evaluationIndices = permutationVector(nTraining + 1:end);

probeModel.treeCount = treeCount;
probeModel.trees = cell(1, treeCount);
nEvaluation = numel(evaluationIndices);
nTrainingActual = numel(trainingIndices);

trainingFeatureMatrix = zeros(nTrainingActual, size(Xall, 2));
trainingTargetVector = zeros(nTrainingActual, 1);
for rowIndex = 1:nTrainingActual
    sourceRowIndex = trainingIndices(rowIndex);
    trainingFeatureMatrix(rowIndex, :) = Xall(sourceRowIndex, :);
    trainingTargetVector(rowIndex) = yall(sourceRowIndex);
end

for treeIndex = 1:treeCount
    probeModel.trees{treeIndex} = grow_decision_tree( ...
        trainingFeatureMatrix, trainingTargetVector, nTrainingActual, 1, inf);
end

evaluationFeatureMatrix = zeros(nEvaluation, size(Xall, 2));
evaluationTargetVector = zeros(nEvaluation, 1);
for rowIndex = 1:nEvaluation
    sourceRowIndex = evaluationIndices(rowIndex);
    evaluationFeatureMatrix(rowIndex, :) = Xall(sourceRowIndex, :);
    evaluationTargetVector(rowIndex) = yall(sourceRowIndex);
end

predictedValues = infer_full_forest(probeModel, evaluationFeatureMatrix);
residualDifference = predictedValues - evaluationTargetVector;
residualValue = sqrt(mean(residualDifference .^ 2));

span = max(yall) - min(yall);
if span > 0
    normalizedResidual = residualValue / span;
else
    normalizedResidual = residualValue;
end
end


function predictions = infer_incremental_pool(model, X)
nRows = size(X, 1);
accumulatedPredictions = zeros(model.treeCount, nRows);
for treeIndex = 1:model.treeCount
    currentTree = model.trees{treeIndex};
    treePredictions = predict(currentTree, X);
    accumulatedPredictions(treeIndex, :) = treePredictions';
end
predictions = mean(accumulatedPredictions, 1)';
end


function net = mlp_initialize(featureDim, hiddenSizes)
layerSizes = [featureDim, hiddenSizes, 1];
nLayers = numel(layerSizes) - 1;
net.W = cell(1, nLayers);
net.b = cell(1, nLayers);
net.mW = cell(1, nLayers);
net.vW = cell(1, nLayers);
net.mb = cell(1, nLayers);
net.vb = cell(1, nLayers);
for layerIndex = 1:nLayers
    fanIn = layerSizes(layerIndex);
    fanOut = layerSizes(layerIndex + 1);
    initializationScale = sqrt(2 / fanIn);
    net.W{layerIndex} = randn(fanOut, fanIn) * initializationScale;
    net.b{layerIndex} = zeros(fanOut, 1);
    net.mW{layerIndex} = zeros(fanOut, fanIn);
    net.vW{layerIndex} = zeros(fanOut, fanIn);
    net.mb{layerIndex} = zeros(fanOut, 1);
    net.vb{layerIndex} = zeros(fanOut, 1);
end
net.step = 0;
net.rate = 0.003;
net.nLayers = nLayers;
end


function net = mlp_set_normalization(net, X, y)
net.xMean = mean(X, 1);
net.xStd = std(X, 0, 1) + 1e-8;
net.yMean = mean(y);
net.yStd = std(y) + 1e-8;
end


function [outValue, cache] = mlp_forward(net, X)
cache.activations = cell(1, net.nLayers + 1);
cache.activations{1} = X;
for layerIndex = 1:net.nLayers
    W = net.W{layerIndex};
    b = net.b{layerIndex};
    previousActivation = cache.activations{layerIndex};
    linearOutput = W * previousActivation + b;
    if layerIndex < net.nLayers
        activationOutput = tanh(linearOutput);
    else
        activationOutput = linearOutput;
    end
    cache.activations{layerIndex + 1} = activationOutput;
end
outValue = cache.activations{end};
end


function net = mlp_backward(net, cache, target)
nLayers = net.nLayers;
nSamples = size(cache.activations{1}, 2);
dZ = cell(1, nLayers);
for layerIndex = nLayers:-1:1
    if layerIndex == nLayers
        outputResidual = cache.activations{nLayers + 1} - target;
        dZ{layerIndex} = outputResidual / nSamples;
    else
        Wnext = net.W{layerIndex + 1};
        dA = Wnext' * dZ{layerIndex + 1};
        activationTerm = cache.activations{layerIndex + 1};
        derivativeTerm = 1 - activationTerm .^ 2;
        dZ{layerIndex} = dA .* derivativeTerm;
    end
end

net.step = net.step + 1;
beta1 = 0.9;
beta2 = 0.999;
epsilonValue = 1e-8;

for layerIndex = 1:nLayers
    dW = dZ{layerIndex} * cache.activations{layerIndex}';
    dB = sum(dZ{layerIndex}, 2);

    net.mW{layerIndex} = beta1 * net.mW{layerIndex} + (1 - beta1) * dW;
    net.vW{layerIndex} = beta2 * net.vW{layerIndex} + (1 - beta2) * dW .^ 2;
    denomW = sqrt(net.vW{layerIndex}) + epsilonValue;
    net.W{layerIndex} = net.W{layerIndex} - net.rate * net.mW{layerIndex} ./ denomW;

    net.mb{layerIndex} = beta1 * net.mb{layerIndex} + (1 - beta1) * dB;
    net.vb{layerIndex} = beta2 * net.vb{layerIndex} + (1 - beta2) * dB .^ 2;
    denomB = sqrt(net.vb{layerIndex}) + epsilonValue;
    net.b{layerIndex} = net.b{layerIndex} - net.rate * net.mb{layerIndex} ./ denomB;
end
end


function net = mlp_train_adam(net, X, y, nEpoch)
normalizedX = (X - net.xMean) ./ net.xStd;
normalizedY = (y - net.yMean) / net.yStd;
transposedX = normalizedX';
transposedY = normalizedY';
nSamples = size(transposedX, 2);
miniBatchSize = 32;
for epochIndex = 1:nEpoch
    shuffledOrder = randperm(nSamples);
    nBatches = ceil(nSamples / miniBatchSize);
    for batchIndex = 1:nBatches
        batchStart = (batchIndex - 1) * miniBatchSize + 1;
        batchEnd = min(batchIndex * miniBatchSize, nSamples);
        batchIndices = shuffledOrder(batchStart:batchEnd);
        batchFeatures = transposedX(:, batchIndices);
        batchTargets = transposedY(:, batchIndices);
        [~, cacheData] = mlp_forward(net, batchFeatures);
        net = mlp_backward(net, cacheData, batchTargets);
    end
end
end


function predictions = mlp_predict(net, X)
normalizedX = (X - net.xMean) ./ net.xStd;
transposedX = normalizedX';
[forwardOutput, ~] = mlp_forward(net, transposedX);
denormalizedOutput = forwardOutput' * net.yStd + net.yMean;
predictions = denormalizedOutput;
end


function [Xboot, yboot] = bootstrap_resample(X, y, ratio)
nSamples = size(X, 1);
resampleCount = round(ratio * nSamples);
if resampleCount < 1
    resampleCount = 1;
end
resampleIndices = randi(nSamples, resampleCount, 1);
Xboot = zeros(resampleCount, size(X, 2));
yboot = zeros(resampleCount, 1);
for rowIndex = 1:resampleCount
    sourceRowIndex = resampleIndices(rowIndex);
    Xboot(rowIndex, :) = X(sourceRowIndex, :);
    yboot(rowIndex) = y(sourceRowIndex);
end
end


function [Xheld, yheld] = hold_out_partition(X, y, ratio)
nSamples = size(X, 1);
permutationVector = randperm(nSamples);
holdOutCount = round(ratio * nSamples);
if holdOutCount < 1
    holdOutCount = 1;
end
holdOutIndices = permutationVector(1:holdOutCount);
Xheld = zeros(holdOutCount, size(X, 2));
yheld = zeros(holdOutCount, 1);
for rowIndex = 1:holdOutCount
    sourceRowIndex = holdOutIndices(rowIndex);
    Xheld(rowIndex, :) = X(sourceRowIndex, :);
    yheld(rowIndex) = y(sourceRowIndex);
end
end


function outMatrix = column_normalize(matrixIn)
nRows = size(matrixIn, 1);
nCols = size(matrixIn, 2);
outMatrix = zeros(nRows, nCols);
for columnIndex = 1:nCols
    columnValues = matrixIn(:, columnIndex);
    columnMaximum = max(columnValues);
    columnMinimum = min(columnValues);
    columnSpan = columnMaximum - columnMinimum;
    if columnSpan < 1e-12
        columnSpan = 1;
    end
    for rowIndex = 1:nRows
        outMatrix(rowIndex, columnIndex) = ...
            (matrixIn(rowIndex, columnIndex) - columnMinimum) / columnSpan;
    end
end
end


function distanceMatrix = pairwise_distance(A, B)
nA = size(A, 1);
nB = size(B, 1);
distanceMatrix = zeros(nA, nB);
for rowIndexA = 1:nA
    for rowIndexB = 1:nB
        differenceVector = A(rowIndexA, :) - B(rowIndexB, :);
        squaredElements = differenceVector .^ 2;
        squaredSum = sum(squaredElements);
        distanceMatrix(rowIndexA, rowIndexB) = sqrt(squaredSum);
    end
end
end


function valid = validate_feature_dim(X, expectedDim)
actualDim = size(X, 2);
valid = (actualDim == expectedDim);
end


function Xshuffled = shuffle_rows(X)
nRows = size(X, 1);
permutationVector = randperm(nRows);
Xshuffled = zeros(nRows, size(X, 2));
for rowIndex = 1:nRows
    sourceRowIndex = permutationVector(rowIndex);
    Xshuffled(rowIndex, :) = X(sourceRowIndex, :);
end
end


function normValue = vector_squared_norm(inputVector)
squaredElements = inputVector .^ 2;
normValue = sum(squaredElements);
end


function spanValue = vector_span(inputVector)
maximumValue = max(inputVector);
minimumValue = min(inputVector);
spanValue = maximumValue - minimumValue;
end


function meanValue = vector_mean(inputVector)
nElements = numel(inputVector);
if nElements == 0
    meanValue = NaN;
    return;
end
accumulator = 0;
for elementIndex = 1:nElements
    accumulator = accumulator + inputVector(elementIndex);
end
meanValue = accumulator / nElements;
end


function stdValue = vector_standard_deviation(inputVector)
nElements = numel(inputVector);
if nElements < 2
    stdValue = 0;
    return;
end
meanValue = vector_mean(inputVector);
accumulator = 0;
for elementIndex = 1:nElements
    deviationValue = inputVector(elementIndex) - meanValue;
    squaredDeviation = deviationValue .^ 2;
    accumulator = accumulator + squaredDeviation;
end
varianceValue = accumulator / (nElements - 1);
stdValue = sqrt(varianceValue);
end


function result = safe_divide(numeratorValue, denominatorValue)
if denominatorValue == 0
    result = 0;
    return;
end
result = numeratorValue / denominatorValue;
end


function clippedValue = clip_to_interval(valueIn, lowerLimit, upperLimit)
if valueIn < lowerLimit
    clippedValue = lowerLimit;
    return;
end
if valueIn > upperLimit
    clippedValue = upperLimit;
    return;
end
clippedValue = valueIn;
end


function checkResult = check_empty(matrixIn)
checkResult = isempty(matrixIn);
end


function positiveFlag = assert_positive(valueIn)
positiveFlag = valueIn > 0;
end


function sizeMatchFlag = is_valid_size(matrixA, matrixB)
sizeA = size(matrixA, 1);
sizeB = size(matrixB, 1);
sizeMatchFlag = (sizeA == sizeB);
end


function cellArrayCopy = duplicate_cell_array(inputCellArray)
nElements = numel(inputCellArray);
cellArrayCopy = cell(1, nElements);
for elementIndex = 1:nElements
    cellArrayCopy{elementIndex} = inputCellArray{elementIndex};
end
end


function numericValue = extract_numeric_from_cell(inputCell, cellIndex)
contentValue = inputCell{cellIndex};
if isnumeric(contentValue)
    numericValue = contentValue;
else
    numericValue = NaN;
end
end


function outputVector = create_index_vector(nElements)
outputVector = zeros(1, nElements);
for index = 1:nElements
    outputVector(index) = index;
end
end


function outputVector = reverse_index_vector(inputVector)
nElements = numel(inputVector);
outputVector = zeros(1, nElements);
for index = 1:nElements
    outputVector(index) = inputVector(nElements - index + 1);
end
end