python extract_feature/gather_video_paths.py
python extract_feature/extract.py --dataflow --csv /output/csv/slowfast_info.csv --batch_size 25 --num_decoding_thread 4 --clip_len 3/2 TEST.CHECKPOINT_FILE_PATH /models/SLOWFAST_8x8_R50.pkl
