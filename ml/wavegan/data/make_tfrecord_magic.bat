CALL activate tensorflow
python make_tfrecord.py ^
	.\Preprocessed\Magic ^
	.\Final_Datasets\Magic ^
    --first_only ^
	--name train ^
	--ext wav ^
	--fs 16000 ^
	--nshards 128 ^
	--slice_len 1.5
