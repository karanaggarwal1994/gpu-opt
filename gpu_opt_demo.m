function gpu_opt_demo()

if ~exist('vistaRootPath.m','file')
    disp('Vistasoft package either not installed or not on matlab path.')
    error('Please, download it from https://github.com/vistalab/vistasoft');
end

if ~exist('mbaComputeFibersOutliers','file')
    disp('ERROR: mba package either not installed or not on matlab path.')
    error('Please, download it from https://github.com/francopestilli/mba')
end

if ~exist('feDemoDataPath.m','file')
    disp('ERROR: demo dataset either not installed or not on matlab path.')
    error('Please, download it from http://purl.dlib.indiana.edu/iusw/data/2022/20995/Demo_Data_for_Multidimensional_Encoding_of_Brain_Connectomes.tar.gz ')
end

dwiFile       = fullfile(feDemoDataPath('STN','sub-FP','dwi'),'run01_fliprot_aligned_trilin.nii.gz');
dwiFileRepeat = fullfile(feDemoDataPath('STN','sub-FP','dwi'),'run02_fliprot_aligned_trilin.nii.gz');
t1File        = fullfile(feDemoDataPath('STN','sub-FP','anatomy'),  't1.nii.gz');

fgFileName = fullfile(feDemoDataPath('STN','sub-FP','tractography'),'run01_fliprot_aligned_trilin_csd_lmax10_wm_SD_PROB-NUM01-500000.tck');
feFileName = 'LiFE_build_model_demo_STN_FP_CSD_PROB';

L = 360;
Niter = 501;

fe = feConnectomeInit(dwiFile,fgFileName,feFileName,[],dwiFileRepeat,t1File,L,[1,0]);
fe = feSet(fe,'fit',feFitModel_gpu_opt(feGet(fe,'model'),feGet(fe,'dsigdemeaned'),'bbnnls',Niter,'preconditioner'));

rmse1 = feGet(fe, 'total rmse');
rmse2 = feGetRep(fe, 'total rmse');
weights = feGet(fe, 'fiber weights');
wnorm = sum(weights);
nnz = sum(weights~=0);

fprintf('*************************************************\n');
fprintf('RMSE1 (train error): %f\n', rmse1);
fprintf('RMSE2 (cross validation error): %f\n', rmse2);
fprintf('wnorm (summed weights): %f\n', wnorm);
fprintf('nnz (number of non zero weihgts): %d\n', nnz);
fprintf('*************************************************\n');

end