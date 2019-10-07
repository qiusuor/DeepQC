# Simple example.

## Just run:
`
bash run.sh
`
## Or you can test step by step:

### Test whether the pre-compiled excutable binary works
`
../c++/DeepQC --usage
`

If it doesn't work, you may need to configure Libtorch environment and re-compile DeepQC in `c++` folder.

### Compression
`
../c++/DeepQC c ../pre_trained_models/Hiseq2000_len_101_1.stat_info ../pre_trained_models/Hiseq2000_len_101_1.pt test.qs test.bin 20 2>/dev/null
`

### Decompression
`
../c++/DeepQC x ../pre_trained_models/Hiseq2000_len_101_1.stat_info ../pre_trained_models/Hiseq2000_len_101_1.pt test.bin test.rec.qs 20 2>/dev/null
`

### Check MD5

`
md5sum test.qs test.rec.qs
`
