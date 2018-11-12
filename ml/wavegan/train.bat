python train_wavegan.py train .\train ^
--data_dir .\data\Final_Datasets\All ^
--data_first_window ^
--use_extra_uncond_loss ^
--wavegan_kernel_len 8 ^
--wavegan_genr_upsample zeros ^
--train_batch_size 50 ^
--wavegan_disc_nupdates 1 ^
--train_save_secs 900 ^
--use_pixel_norm ^
--use_disc_layernorm ^
--train_summary_secs 15

REM --use_extra_uncond_loss ^
REM --wavegan_loss dcgan ^
REM --wavegan_batchnorm ^
REM --wavegan_disc_nupdates 1 ^
REM --wavegan_genr_upsample nn ^
REM --wavegan_kernel_len 8 ^
REM --wavegan_disc_phaseshuffle 0 ^
REM --wavegan_genr_pp ^