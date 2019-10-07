set -x -e
../c++/DeepQC --usage

../c++/DeepQC c ../pre_trained_models/Hiseq2000_len_101_1.stat_info ../pre_trained_models/Hiseq2000_len_101_1.pt test.qs test.bin 20 2>/dev/null

../c++/DeepQC x ../pre_trained_models/Hiseq2000_len_101_1.stat_info ../pre_trained_models/Hiseq2000_len_101_1.pt test.bin test.rec.qs 20 2>/dev/null

md5sum test.qs test.rec.qs

