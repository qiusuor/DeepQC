#pragma once
//
// Created by qiusuo on 2019/6/2.
//

#ifndef LSTM_COMPRESSION_MEM_H
#define LSTM_COMPRESSION_MEM_H

#include <cstdint>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "arithmatic_coder.h"



using namespace std;


SYMBOL probabilities[num_max_batch_size][64]{
        {
                { 'B',  0,  1  },
                { 'I',  1,  2  },
                { 'L',  2,  4  },
                { ' ',  4,  5  },
                { 'G',  5,  6  },
                { 'A',  6,  7  },
                { 'T',  7,  8  },
                { 'E',  8,  9  },
                { 'S',  9,  10 },
                { '\0', 10, 11  }

        }
};
uint64_t num_reads_per_batch = 300;
extern uint64_t num_reads_per_batch;
//const uint64_t binary_map_size = 536870912;//512M

const uint64_t one_out_size=536870912*2/num_max_batch_size;

char characters[128];
SYMBOL st_pro[128];
uint32_t st_num;
uint32_t num_ch=38;
uint64_t num_max_seq_len = 101;
float pd[1000][128];


char char2int[128];
//__uint32_t statisticCum[35];

void config_params(const char* stat_info){
    ifstream st(stat_info);
    st>>num_ch;
    for (int l = 0; l < num_ch; ++l) {
        st>>characters[l];
    }
    st>>num_max_seq_len;

    for(uint i=0;i<num_ch;i++){
//			statisticCum[i]=i+1;
        char2int[characters[i]]=i;
    }
    st>>st_num;
    char t;
    int *pos=new int[st_num];
    for (int i = 0; i <st_num ; ++i) {
        st>>t;
        pos[i]=char2int[t];
//        cout<<int(char2int[t])<<endl;
        st_pro[char2int[t]].c=t;
    }
    for (int j = 0; j < st_num; ++j) {
        st>>st_pro[pos[j]].high;
    }
    st_pro[pos[0]].low=0;
    st_pro[pos[0]].scale=st_pro[pos[st_num-1]].high;
    for (int k = 1; k < st_num; ++k) {
        st_pro[pos[k]].low=st_pro[pos[k-1]].high;
        st_pro[pos[k]].scale=st_pro[pos[st_num-1]].high;
    }
    uint num_lines;
    st>>num_lines;

    for(int i=0;i<num_max_seq_len;i++)
        for (int m = 0; m < num_ch; ++m) {
            st>>pd[i][m];
            pd[i][m]/=num_lines;
        }


    delete [] pos;

    st.close();

}
class input {
public:
	char *content;
	uint64_t idx_file = 0;
	ifstream inputFile;
	uint64_t true_batch_size = 0;
	at::Tensor data=torch::zeros({num_max_batch_size},torch::TensorOptions().dtype(torch::kInt8));
	char one_col[num_max_batch_size];
	uint64_t col_index = 0;
	input(string fileName) {
		content = (char*)malloc(num_max_seq_len * num_reads_per_batch*num_max_batch_size);
		inputFile.open(fileName);
	}
	void refresh() {
		string line;
		idx_file = 0;
		while (idx_file < num_reads_per_batch * num_max_batch_size&& inputFile >> line) {
			for (uint64_t i = 0; i < num_max_seq_len; i++)
			content[idx_file*num_max_seq_len + i] = char2int[line[i]];
			idx_file++;
		}
		true_batch_size = idx_file / num_reads_per_batch;
		if (idx_file%num_reads_per_batch)
		{
			true_batch_size++;
		}
		col_index=0;

	}
	at::Tensor& get_col(bool *is_good) {
		if (col_index>= num_reads_per_batch * num_max_seq_len)
		{
			*is_good= false;
			return data;
		}
		*is_good= true;
		for (uint64_t i = 0; i < true_batch_size; i++) {
			one_col[i]=content[i*num_reads_per_batch * num_max_seq_len + col_index];
		}
		mempcpy(data.data_ptr(),one_col,num_max_batch_size);
		col_index++;
		return data;
	}

	~input() {
		inputFile.close();
		free(content);
	}

};



class tempDecoder{
public:

	vector<char*> outputs;
	char one_col[num_max_batch_size];
	mem *mems;
	uint32_t true_batch_size;

	tempDecoder(mem *mems,__uint32_t true_batch_size){

		this->mems=mems;
		this->true_batch_size= true_batch_size;
		outputs.resize(true_batch_size);

		for(uint64_t i=0;i<true_batch_size;i++){
			outputs[i]=(char *)malloc(one_out_size);
			mems[i].start=0;
		}
	}
	~tempDecoder(){
		for(uint64_t i=0;i<true_batch_size;i++){
			free(outputs[i]);
		}
	}
	void writeQs(ofstream &out_file,uint32_t num_rows){

		for(int i=0;i<true_batch_size-1;i++){
			for (int k=0;k<num_reads_per_batch;k++){
				for (int j = 0; j <num_max_seq_len ; j++) {
					out_file<<outputs[i][k*num_max_seq_len+j];
				}
				out_file<<endl;
			}


		}
		uint32_t remin=num_rows%num_reads_per_batch;
		if(remin==0)
		    remin=num_reads_per_batch;
        for (int k=0;k<remin;k++){
            for (int j = 0; j <num_max_seq_len ; ++j) {
                out_file<<outputs[true_batch_size-1][k*num_max_seq_len+j];
            }
            out_file<<endl;
        }

	}
	at::Tensor decodeFirst(SYMBOL *statisticCum,__uint32_t true_batch_size,Coder *coders,BitIO* ios){
	    SYMBOL s;
	    s.scale=st_pro[0].scale;
	    uint32_t count;
		for(uint64_t i=0;i<true_batch_size;i++){
			ios[i].initialize_input_bitstream();
			coders[i].initialize_arithmetic_decoder( mems[i] ,ios[i]);
		    count=coders[i].get_current_count(&s);

			for (int j = num_ch-1; j >=0; --j) {
				if (count >= statisticCum[j].low && count < statisticCum[j].high) {
					coders[i].remove_symbol_from_stream(mems[i],&statisticCum[j],ios[i]);
                    one_col[i] = j;
                    outputs[i][0]=characters[j];
					break;
				}
			}

        }
		at::Tensor data=torch::zeros({num_max_batch_size},torch::TensorOptions().dtype(torch::kInt8));
		mempcpy(data.data_ptr(),one_col,num_max_batch_size);

		return data;
	}

};
class fileToMem{
public:
	char *binaryMap= nullptr;
	__uint32_t true_batch_size;
	__uint32_t total_size;
	uint32_t num_rows;
	mem *mems= nullptr;
	vector<uint32_t> start_pos;
	uint32_t file_pos;
	ifstream inFile;
	bool refresh(){
		inFile.seekg(file_pos+0);
		inFile.read((char*)&(this->total_size),4);
		inFile.read((char*)&(this->true_batch_size),4);
		inFile.read((char*)&(this->num_rows),4);
		if(binaryMap!= nullptr)
			free(binaryMap);
		binaryMap = (char *)malloc(total_size);

		inFile.seekg(file_pos+0);
		inFile.read((char*)binaryMap,total_size);

		inFile.seekg(file_pos+12);
		start_pos.resize(0);
		for(uint64_t i=0;i<true_batch_size;i++){
			uint32_t tm;
			inFile.read((char*)&tm,4);
			start_pos.push_back(tm);
			mems[i].start=0;

		}
		for(uint64_t i=0;i<true_batch_size-1;i++){
			mems[i].true_size=start_pos[i+1]-start_pos[i];
			memcpy(mems[i].data,binaryMap+start_pos[i],start_pos[i+1]-start_pos[i]);
		}
		mems[true_batch_size-1].true_size=total_size-start_pos[true_batch_size-1];
		memcpy(mems[true_batch_size-1].data,binaryMap+start_pos[true_batch_size-1],total_size-start_pos[true_batch_size-1]);
		file_pos+=this->total_size;
		return inFile.good();
	}
	fileToMem(string fileName) {

		inFile.open(fileName,ios::binary);
		file_pos=0;
		mems=new mem[num_max_batch_size];

	}
	~fileToMem(){
		delete []mems;
		if(binaryMap!= nullptr)
			free(binaryMap);
		inFile.close();
	}
};
class memToFile {
public:
	char *binaryMap;
	__uint32_t true_batch_size;
	__uint32_t total_size;
	uint32_t num_rows;

	vector<uint32_t> start_pos;
	memToFile(uint32_t true_batch_size,vector<uint32_t>& out_size,uint32_t num_rows) {
        if(true_batch_size){
            this->true_batch_size=true_batch_size;
            this->num_rows=num_rows;
            start_pos.resize(true_batch_size);
            start_pos[0]=(true_batch_size+3)*4;
            for (int i = 1; i < true_batch_size; ++i) {
                start_pos[i]=start_pos[i-1]+out_size[i-1];
            }
            this->total_size=start_pos[true_batch_size-1]+out_size[true_batch_size-1];
            binaryMap = (char *)malloc(total_size);
        }

	}

	void writeToMem(mem *mems,vector<uint32_t>& out_size){

		uint32_t *ptr=(uint32_t*)binaryMap;
		*ptr++=total_size;
		*ptr++=true_batch_size;
        *ptr++=num_rows;
		for (int i = 0; i < true_batch_size; ++i) {
			*ptr++=start_pos[i];
		}
		unsigned char *cptr=(unsigned char *)ptr;
		for (int i = 0; i < true_batch_size; ++i) {
			memcpy(cptr,mems[i].data,out_size[i]);
			cptr+=out_size[i];
		}
		cout<<"current block size: "<<total_size<<endl;
	}
	void writeToFile(ofstream & outFile){

		outFile.write((char*)binaryMap,total_size);
		outFile.flush();
	}
};


#endif //LSTM_COMPRESSION_MEM_H

