python extract_feature/gather_video_paths.py
python extract_feature/extract.py --dataflow --csv /output/csv/slowfast_info.csv --batch_size 20 --num_decoding_thread 16 --clip_len 2/3 TEST.CHECKPOINT_FILE_PATH /models/SLOWFAST_8x8_R50.pkl
