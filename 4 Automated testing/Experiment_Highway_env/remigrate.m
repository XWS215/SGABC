function remigrate()
%REMIGRATE restore original model from migration backup and then migrate again

%restore old model
backupfolder = '.\Migrated\version_[]_153333' ;
backupmodel  = 'Experiment_Highway_env_[]_153333_cs' ;
backupfile   = '.\Migrated\version_[]_153333\Experiment_Highway_env_[]_153333_cs.slx' ;
srcfolder    = '.' ;
srcmodel     = 'Experiment_Highway_env_cs' ;
modelreferencing = 0 ;
mbxutils.backupModel(backupfolder,backupmodel,srcfolder,srcmodel); 
chdir('.');
open_system('Experiment_Highway_env_cs');

%migrate restored model
migrate_all(srcmodel,modelreferencing,backupfile);
end
