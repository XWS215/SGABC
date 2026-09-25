% ============================================================
% PreScan Side Parameters
% The PreScan side parameters are applied by reading the data
% model of the experiment into a MATLAB structure, modifying
% the relevant fields, and then passing the modified structure
% directly to the PreScan run function. This keeps each test
% case isolated and avoids overwriting the original data model.
%
% CarSim Side Parameters
% The CarSim side parameters are passed through the Simulink
% workspace by means of SimulationInput variables. These
% variables are read by the CarSim interface blocks inside the
% Simulink model at initialization time.
%
% Initialization Method and Run
% Each test case is executed by calling the PreScan run function
% with the modified data model structure and the desired stop
% time. The returned simulation output is stored for later
% analysis. A timeout mechanism is included to abort the test
% run if the total elapsed time exceeds a predefined limit.
% ============================================================

MODEL_NAME  = 'Experiment_Highway_env_cs';
PB_FILE     = 'Experiment_Highway_env.pb';
OUTPUT_DIR  = fullfile(pwd, 'TestResults');
TIMEOUT_SEC = 600;

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

load_system(MODEL_NAME);

testCases = loadTestCases();
numCases  = size(testCases, 1);

testStartTime = tic;
aborted       = false;

for iCase = 1:numCases

    if toc(testStartTime) > TIMEOUT_SEC
        fprintf('Timeout of %d seconds exceeded. Aborting.\n', TIMEOUT_SEC);
        aborted = true;
        break;
    end

    tc        = testCases(iCase, :);
    EGO_Speed = tc(1) / 3.6;
    LV_Speed  = tc(2) / 3.6;
    FV_Speed  = tc(3) / 3.6;
    FFV_Speed = tc(4) / 3.6;
    LV_Y      = tc(5);
    LV_X      = tc(6);
    FV_X      = tc(7);
    FFV_X     = tc(8);
    FFV_ACC   = tc(9);
    LV_Cutin  = tc(10);
    FV_Cutin  = tc(11);
    MU        = tc(12);
    rainfall  = tc(13);

    models = prescan.experiment.readDataModels(PB_FILE);

    for k = 1:numel(models.worldmodel.object)

        obj  = models.worldmodel.object{k};
        name = obj.name;

        switch name

            case 'EG'
                obj.speed = EGO_Speed;

            case 'LV'
                obj.speed = LV_Speed;
                obj.pose.position.x = LV_X;
                obj.pose.position.y = LV_Y;

            case 'FV'
                obj.speed = FV_Speed;
                obj.pose.position.x = FV_X;

            case 'FFV'
                obj.speed = FFV_Speed;
                obj.pose.position.x = FFV_X;

        end

    end

    if isfield(models.worldmodel, 'weather')
        models.worldmodel.weather.rainfall = rainfall;
    end

    in = Simulink.SimulationInput(MODEL_NAME);

    in = in.setVariable('EGO_Speed', EGO_Speed);
    in = in.setVariable('LV_Speed',  LV_Speed);
    in = in.setVariable('FV_Speed',  FV_Speed);
    in = in.setVariable('FFV_Speed', FFV_Speed);
    in = in.setVariable('FFV_ACC',   FFV_ACC);
    in = in.setVariable('LV_Y',      LV_Y);
    in = in.setVariable('LV_Cutin',  LV_Cutin);
    in = in.setVariable('FV_Cutin',  FV_Cutin);
    in = in.setVariable('MU',        MU);
    in = in.setVariable('rainfall',  rainfall);

    try

        simOut = prescan.experiment.runWithDataModels(models, ...
            'StopTime', '15');

        resultFile = fullfile(OUTPUT_DIR, ...
            sprintf('ScenarioA_Case%02d.mat', iCase));
        save(resultFile, 'simOut', 'tc');

    catch ME

        failLog = fullfile(OUTPUT_DIR, 'failed_cases.txt');
        fid = fopen(failLog, 'a');
        fprintf(fid, 'Case %02d: %s\n', iCase, ME.message);
        fclose(fid);

    end

end

close_system(MODEL_NAME, 0);

if ~aborted
    fprintf('All test cases completed within the time limit.\n');
end