#include "men.h"
#include <ctime>
#include <vector>
#include "arithmatic_coder.h"
#include <stdlib.h>

using namespace std;

using namespace torch;


void fill_pros(int32_t *rel_out){
    for(uint i=0;i<num_max_batch_size;i++){
        probabilities[i][0].c=characters[0];
        probabilities[i][0].low=0;
        probabilities[i][0].high=rel_out[i*num_ch+0];
        probabilities[i][0].scale=rel_out[i*num_ch+num_ch-1];
        for(uint j=1;j<num_ch;j++){
            probabilities[i][j].c=characters[j];
            probabilities[i][j].low=rel_out[i*num_ch+j-1];
            probabilities[i][j].high=rel_out[i*num_ch+j];
            probabilities[i][j].scale=rel_out[i*num_ch+num_ch-1];

        }
    }
}



int main(int argc, const char* argv[]) {
    if(argc<7){
        cout<<"usage: DeepQC <c|x> stat_file model.pt input_file output_file alpha\n";
        cout<<"\n<c|x>: c for compress; x for decompress\n";
        cout<<"alpha: indicates how many reads concatenated together in a row of a parallel group. \n    we suggest set alpha to 20 when the quality score file size smaller than 100MB, 1000 when larger than 1GB.\n";
        return 0;
    }
    num_reads_per_batch=atoi(argv[6]);

    config_params(argv[2]);
    BitIO *ios=new BitIO[num_max_batch_size];
    Coder *coders=new Coder[num_max_batch_size];
    mem *mems=new mem[num_max_batch_size];




    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type, 0);

    torch::DeviceType cpu_type;
    cpu_type = torch::kCPU;
    torch::Device cpu(cpu_type, 0);
    auto module = torch::jit::load(argv[3]);
    module.to(device);

    at::Tensor pos=torch::zeros({num_ch});
    pos=pos.contiguous();
    auto pos_ptr= pos.data<float>();

    if(argv[1][0]=='c'){
        ofstream outFile(argv[5],ios::binary);
        input a(argv[4]);

        while(a.inputFile.good()){
            a.refresh();
            bool is_good;

            auto data=a.get_col(&is_good);
            if(a.true_batch_size){
                auto data_pre= data.data<signed char>();


                for(uint64_t i=0;i<a.true_batch_size;i++) {
                    mems[i].start = 0;
                    ios[i].initialize_output_bitstream();
                    coders[i].initialize_arithmetic_encoder(i);
                    coders[i].encode_symbol(mems[i], &st_pro[data_pre[i]], ios[i], i);

                }
                auto h=torch::zeros({2,num_max_batch_size,128}).to(device);
                auto c=torch::zeros({2,num_max_batch_size,128}).to(device);
                auto mem_0=torch::zeros({num_max_batch_size,15,1}).to(device);
                auto mem_1=torch::zeros({num_max_batch_size,15,1}).to(device);
                for(uint i=0;i<num_ch;i++){
                    *(pos_ptr+i)=pd[0][i];
                }
                auto output = module.forward({data.to(torch::kFloat).reshape({1,num_max_batch_size,1}).to(device),h,c,pos.to(device),mem_0,mem_1}).toTuple();
                h=output->elements()[1].toTensor().detach();
                c=output->elements()[2].toTensor().detach();
                mem_0=output->elements()[3].toTensor().detach();
                mem_1=output->elements()[4].toTensor().detach();

                auto b=output->elements()[0].toTensor().to(cpu).detach();
                b=b.contiguous();


                auto rel_out=b.data<int32_t >();
                fill_pros(rel_out);
                int f= 0;
                while (is_good){
                    f++;
                    data=a.get_col(&is_good).detach();
                    data_pre= data.data<signed char>();
                    for(uint i=0;i<num_ch;i++){
                        *(pos_ptr+i)=pd[f%(num_max_seq_len)][i];
                    }
                    if((f)%num_max_seq_len){
                        for(uint64_t i=0;i<a.true_batch_size;i++){
                            if((i*num_max_seq_len*num_reads_per_batch+f)/num_max_seq_len<a.idx_file)
                                coders[i].encode_symbol(mems[i],&probabilities[i][data_pre[i]],ios[i],i);
                        }
                    }

                    else{
                        for(uint64_t i=0;i<a.true_batch_size;i++){
                            if((i*num_max_seq_len*num_reads_per_batch+f)/num_max_seq_len<a.idx_file)
                                coders[i].encode_symbol(mems[i],&st_pro[data_pre[i]],ios[i],i);
                        }
                    }
                    output = module.forward({data.to(torch::kFloat).reshape({1,num_max_batch_size,1}).to(device),h,c,pos.to(device),mem_0,mem_1}).toTuple();
                    if((f+1)%num_max_seq_len){
                        h=output->elements()[1].toTensor().detach();
                        c=output->elements()[2].toTensor().detach();
                        mem_0=output->elements()[3].toTensor().detach();
                        mem_1=output->elements()[4].toTensor().detach();
                    }
                    else{
                        h=torch::zeros({2,num_max_batch_size,128}).to(device);
                        c=torch::zeros({2,num_max_batch_size,128}).to(device);
                        mem_0=torch::zeros({num_max_batch_size,15,1}).to(device);
                        mem_1=torch::zeros({num_max_batch_size,15,1}).to(device);
                    }

                    b=output->elements()[0].toTensor().to(cpu).detach();
                    b=b.contiguous();
                    rel_out=b.data<int32_t >();
                    fill_pros(rel_out);
                }

                vector<uint32_t> out_size(a.true_batch_size);
                for (int k = 0; k <a.true_batch_size ; ++k) {
                    coders[k].flush_arithmetic_encoder(mems[k],ios[k],k);
                    ios[k].flush_output_bitstream(mems[k]);
                    out_size[k]=mems[k].true_size;
                }

                memToFile mem(a.true_batch_size,out_size,a.idx_file);
                mem.writeToMem(mems,out_size);
                mem.writeToFile(outFile);
                outFile.flush();
            }


        }
        outFile.close();
        cout<<"compress finish"<<endl;
    }
    else if(argv[1][0]=='x'){
        fileToMem fm(argv[4]);
        ofstream outfile(argv[5]);
        while (fm.refresh()){

            tempDecoder td(fm.mems,fm.true_batch_size);
            char one_col[num_max_batch_size];
            auto data=td.decodeFirst(st_pro,fm.true_batch_size,coders,ios);
            auto h=torch::zeros({2,num_max_batch_size,128}).to(device);
            auto c=torch::zeros({2,num_max_batch_size,128}).to(device);
            auto mem_0=torch::zeros({num_max_batch_size,15,1}).to(device);
            auto mem_1=torch::zeros({num_max_batch_size,15,1}).to(device);
            for(uint i=0;i<num_ch;i++){
                *(pos_ptr+i)=pd[0][i];
            }
            auto output = module.forward({data.to(torch::kFloat).reshape({1,num_max_batch_size,1}).to(device),h,c,pos.to(device),mem_0,mem_1}).toTuple();
            h=output->elements()[1].toTensor().detach();
            c=output->elements()[2].toTensor().detach();
            mem_0=output->elements()[3].toTensor().detach();
            mem_1=output->elements()[4].toTensor().detach();

            auto b=output->elements()[0].toTensor().to(cpu).detach();
            b=b.contiguous();
            auto rel_out=b.data<int32_t >();

            fill_pros(rel_out);
            for (uint j=1;j<num_max_seq_len*num_reads_per_batch;j++){
                for(uint i=0;i<num_ch;i++){
                    *(pos_ptr+i)=pd[j%(num_max_seq_len)][i];
                }
                for(uint64_t i=0;i<fm.true_batch_size;i++){
                    if((j)%num_max_seq_len){
                        if((i*num_max_seq_len*num_reads_per_batch+j)/num_max_seq_len<fm.num_rows){
                            uint count=coders[i].get_current_count(&probabilities[i][num_ch-1]);
                            for (int k = num_ch-1; k >=0; --k) {
                                if (count >= probabilities[i][k].low && count < probabilities[i][k].high) {
                                    one_col[i] = k;
                                    coders[i].remove_symbol_from_stream(fm.mems[i],&probabilities[i][k],ios[i]);
                                    td.outputs[i][j]=characters[k];
                                    break;
                                }
                            }
                        }

                    }
                    else{
                        if((i*num_max_seq_len*num_reads_per_batch+j)/num_max_seq_len<fm.num_rows){
                            uint count=coders[i].get_current_count(&st_pro[num_ch-1]);
                            for (int k = num_ch-1; k >=0; --k) {
                                if (count >= st_pro[k].low && count < st_pro[k].high) {
                                    one_col[i] = k;
                                    coders[i].remove_symbol_from_stream(fm.mems[i],&st_pro[k],ios[i]);
                                    td.outputs[i][j]=characters[k];
                                    break;
                                }
                            }
                        }

                    }
                }
                mempcpy(data.data_ptr(),one_col,num_max_batch_size);
                output = module.forward({data.to(torch::kFloat).reshape({1,num_max_batch_size,1}).to(device),h,c,pos.to(device),mem_0,mem_1}).toTuple();
                if((j+1)%num_max_seq_len){
                    h=output->elements()[1].toTensor().detach();
                    c=output->elements()[2].toTensor().detach();
                    mem_0=output->elements()[3].toTensor().detach();
                    mem_1=output->elements()[4].toTensor().detach();
                }
                else{
                    h=torch::zeros({2,num_max_batch_size,128}).to(device);
                    c=torch::zeros({2,num_max_batch_size,128}).to(device);
                    mem_0=torch::zeros({num_max_batch_size,15,1}).to(device);
                    mem_1=torch::zeros({num_max_batch_size,15,1}).to(device);
                }
                b=output->elements()[0].toTensor().to(cpu).detach();
                b=b.contiguous();
                rel_out=b.data<int32_t >();
                fill_pros(rel_out);
            }

            td.writeQs(outfile,fm.num_rows);
        }
        outfile.close();
        cout<<"decompress finish\n";
    }

    delete []coders;
    delete []ios;
    delete []mems;

    return 0;
}

