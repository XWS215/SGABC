function remigrate()
%REMIGRATE restore original model from migration backup and then migrate again

%restore old model
backupfolder = '.\Migrated\version_[]_192916' ;
backupmodel  = 'Experiment_Intersection_env_[]_192916_cs' ;
backupfile   = '.\Migrated\version_[]_192916\Experiment_Intersection_env_[]_192916_cs.slx' ;
srcfolder    = '.' ;
srcmodel     = 'Experiment_Intersection_env_cs' ;
modelreferencing = 0 ;
mbxutils.backupModel(backupfolder,backupmodel,srcfolder,srcmodel); 
chdir('.');
open_system('Experiment_Intersection_env_cs');

%migrate restored model
migrate_all(srcmodel,modelreferencing,backupfile);
end
