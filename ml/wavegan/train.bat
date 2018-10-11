CALL activate tensorflow
python train_wavegan.py train .\train ^
--data_dir .\data\Final_Datasets\All ^
--data_first_window ^
--wavegan_genr_pp ^
--train_summary_secs 5

REM --use_extra_uncond_loss ^
REM --wavegan_loss dcgan ^
REM --wavegan_batchnorm ^
REM --wavegan_disc_nupdates 1 ^
REM --wavegan_genr_upsample nn ^
REM --wavegan_kernel_len 8 ^
REM --wavegan_disc_phaseshuffle 0 ^