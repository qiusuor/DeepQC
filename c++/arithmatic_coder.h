//
// Created by qiusuo on 2019/6/25.
//

#ifndef MY_ARITHMATIC_CODER_ARITHMATIC_CODER_H
#define MY_ARITHMATIC_CODER_ARITHMATIC_CODER_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>

using namespace std;
#define MAX_Compressed_size 100000
struct mem{
    char data[MAX_Compressed_size];
    uint32_t start=0;
    uint32_t true_size=0;
    void write( char *buffer, uint64_t size ){
        memcpy(data+start,buffer,size);
        start+=size;
    }
    int read( char *buffer, uint64_t size ){
        if(start+size<true_size){
            memcpy(buffer,data+start,size);
            start=start+size;
            return size;
        }

        memcpy(buffer,data+start,true_size-start);
        uint32_t pre_start=start;
        start=true_size;

        return true_size-pre_start;
    }
};

typedef struct {
    char c;
    uint32_t low;
    uint32_t high;
    uint32_t scale;

} SYMBOL;



#define num_max_batch_size 10240
#define BUFFER_SIZE 256
uint64_t underflow_bits[num_max_batch_size];
//static int out_count=0;
struct BitIO{
    char buffer[ BUFFER_SIZE + 2 ]; /* This is the i/o buffer    */
    char *current_byte;             /* Pointer to current byte   */

    uint32_t output_mask;                /* During output, this byte  */
/* contains the mask that is */
/* applied to the output byte*/
/* if the output bit is a 1  */

    uint32_t input_bytes_left;           /* During input, these three */
    uint32_t input_bits_left;            /* variables keep track of my*/
    uint32_t past_eof;                   /* input state.  The past_eof*/
/* byte comes about because  */
/* of the fact that there is */
/* a possibility the decoder */
/* can legitimately ask for  */
/* more bits even after the  */
/* entire file has been      */
/* sucked dry.               */


/*
 * This routine is called once to initialze the output bitstream.
 * All it has to do is set up the current_byte pointer, clear out
 * all the bits in my current output byte, and set the output mask
 * so it will set the proper bit next time a bit is output.
 */
    void initialize_output_bitstream()
    {
        memset(buffer,0,BUFFER_SIZE + 2);
        current_byte = buffer;
        *current_byte = 0;
        output_mask = 0x80;
    }

/*
 * The output bit routine just has to set a bit in the current byte
 * if requested to.  After that, it updates the mask.  If the mask
 * shows that the current byte is filled up, it is time to go to the
 * next character in the buffer.  If the next character is past the
 * end of the buffer, it is time to flush the buffer.
 */
    void output_bit( mem &stream, int bit )
    {
//        cout<<++out_count<<endl;
        if ( bit )
            *current_byte |= output_mask;
        output_mask >>= 1;
        if ( output_mask == 0 )
        {
            output_mask = 0x80;
            current_byte++;
            if ( current_byte == ( buffer + BUFFER_SIZE ) )
            {
                stream.write( buffer, BUFFER_SIZE);
                current_byte = buffer;
            }
            *current_byte = 0;
        }
    }

/*
 * When the encoding is done, there will still be a lot of bits and
 * bytes sitting in the buffer waiting to be sent out.  This routine
 * is called to clean things up at that point.
 */
    void flush_output_bitstream( mem &stream )
    {
        stream.true_size=stream.start+(size_t)( current_byte - buffer ) + 3;
        stream.write( buffer, (size_t)( current_byte - buffer ) + 3);
        current_byte = buffer;
    }

/*
 * Bit oriented input is set up so that the next time the input_bit
 * routine is called, it will trigger the read of a new block.  That
 * is why input_bits_left is set to 0.
 */
    void initialize_input_bitstream()
    {
        input_bits_left = 0;
        input_bytes_left = 1;
        past_eof = 0;
    }

/*
 * This routine reads bits in from a file.  The bits are all sitting
 * in a buffer, and this code pulls them out, one at a time.  When the
 * buffer has been emptied, that triggers a new file read, and the
 * pointers are reset.  This routine is set up to allow for two dummy
 * bytes to be read in after the end of file is reached.  This is because
 * we have to keep feeding bits into the pipeline to be decoded so that
 * the old stuff that is 16 bits upstream can be pushed out.
 */
    short int input_bit( mem &stream )
    {
        if ( input_bits_left == 0 )
        {
            current_byte++;
            input_bytes_left--;
            input_bits_left = 8;
            if ( input_bytes_left == 0 )
            {
                input_bytes_left = stream.read( buffer, BUFFER_SIZE );
                if ( input_bytes_left == 0 )
                {
                    if ( past_eof )
                    {
                        fprintf( stderr, "Bad input file\n" );
                        exit( -1 );
                    }
                    else
                    {
                        past_eof = 1;
                        input_bytes_left = 2;
                    }
                }
                current_byte = buffer;
            }
        }
        input_bits_left--;
        return ( ( *current_byte >> input_bits_left ) & 1 );
    }




struct Coder{
    uint32_t code;  /* The present input code value       */
    uint32_t low;   /* Start of the current code range    */
    uint32_t high;  /* End of the current code range      */
//    long underflow_bits;             /* Number of underflow bits pending   */

/*
 * This routine must be called to initialize the encoding process.
 * The high register is initialized to all 1s, and it is assumed that
 * it has an infinite string of 1s to be shifted into the lower bit
 * positions when needed.
 */
    void initialize_arithmetic_encoder( int i )
    {
        low = 0;
        high = 0xffffffff;
        underflow_bits[i] = 0;
    }

/*
 * This routine is called to encode a symbol.  The symbol is passed
 * in the SYMBOL structure as a low count, a high count, and a range,
 * instead of the more conventional probability ranges.  The encoding
 * process takes two steps.  First, the values of high and low are
 * updated to take into account the range restriction created by the
 * new symbol.  Then, as many bits as possible are shifted out to
 * the output stream.  Finally, high and low are stable again and
 * the routine returns.
 */
    void encode_symbol( mem &stream, SYMBOL *s ,BitIO &io,int i)
    {
        uint64_t range;
/*
 * These three lines rescale high and low for the new symbol.
 */
        range = (uint64_t) ( high-low ) + 1;
        high = low + (uint32_t)
                (( range * s->high ) / s->scale - 1 );
        low = low + (uint32_t)
                (( range * s->low ) / s->scale );
/*
 * This loop turns out new bits until high and low are far enough
 * apart to have stabilized.
 */
        for ( ; ; )
        {
/*
 * If this test passes, it means that the MSDigits match, and can
 * be sent to the output stream.
 */
            if ( ( high & 0x80000000 ) == ( low & 0x80000000 ) )
            {
//                std::cout<<(high & 0x8000)<<endl;
                io.output_bit( stream, high & 0x80000000 );
                while ( underflow_bits[i] > 0 )
                {
                    io.output_bit( stream, ~high & 0x80000000 );
                    underflow_bits[i]--;
                }
            }
/*
 * If this test passes, the numbers are in danger of underflow, because
 * the MSDigits don't match, and the 2nd digits are just one apart.
 */
            else if ( ( low & 0x40000000 ) && !( high & 0x40000000 ))
            {
                underflow_bits[i] += 1;
                low &= 0x3fffffff;
                high |= 0x40000000;
            }
            else
                return ;
            low <<= 1;
            high <<= 1;
            high |= 1;
        }
    }

/*
 * At the end of the encoding process, there are still significant
 * bits left in the high and low registers.  We output two bits,
 * plus as many underflow bits as are necessary.
 */
    void flush_arithmetic_encoder( mem &stream ,BitIO &io,int i)
    {
        io.output_bit( stream, low & 0x40000000 );
        underflow_bits[i]++;
        while ( underflow_bits[i]-- > 0 )
            io.output_bit( stream, ~low & 0x40000000 );
    }

/*
 * When decoding, this routine is called to figure out which symbol
 * is presently waiting to be decoded.  This routine expects to get
 * the current model scale in the s->scale parameter, and it returns
 * a count that corresponds to the present floating point code:
 *
 *  code = count / s->scale
 */
    uint32_t get_current_count( SYMBOL *s )
    {
        uint64_t range;
        uint32_t count;

        range = (uint64_t) ( high - low ) + 1;
        count = (uint32_t)
                ((((uint64_t) ( code - low ) + 1 ) * s->scale-1 ) / range );
        return( count );
    }

/*
 * This routine is called to initialize the state of the arithmetic
 * decoder.  This involves initializing the high and low registers
 * to their conventional starting values, plus reading the first
 * 16 bits from the input stream into the code value.
 */
    void initialize_arithmetic_decoder( mem &stream ,BitIO &io)
    {
        uint64_t i;

        code = 0;
        for ( i = 0 ; i < 32 ; i++ )
        {
            code <<= 1;
            code += io.input_bit( stream );
        }
        low = 0;
        high = 0xffffffff;
    }

/*
 * Just figuring out what the present symbol is doesn't remove
 * it from the input bit stream.  After the character has been
 * decoded, this routine has to be called to remove it from the
 * input stream.
 */
    void remove_symbol_from_stream( mem &stream, SYMBOL *s,BitIO &io )
    {
        uint64_t range;

/*
 * First, the range is expanded to account for the symbol removal.
 */
        range = (uint64_t)( high - low ) + 1;
        high = low + (uint64_t)
                (( range * s->high ) / s->scale - 1 );
        low = low + (uint64_t)
                (( range * s->low ) / s->scale );
/*
 * Next, any possible bits are shipped out.
 */
        for ( ; ; )
        {
/*
 * If the MSDigits match, the bits will be shifted out.
 */
            if ( ( high & 0x80000000 ) == ( low & 0x80000000 ) )
            {
            }
/*
 * Else, if underflow is threatining, shift out the 2nd MSDigit.
 */
            else if ((low & 0x40000000) == 0x40000000  && (high & 0x40000000) == 0 )
            {
                code ^= 0x40000000;
                low   &= 0x3fffffff;
                high  |= 0x40000000;
            }
/*
 * Otherwise, nothing can be shifted out, so I return.
 */
            else
                return;
            low <<= 1;
            high <<= 1;
            high |= 1;
            code <<= 1;
            code += io.input_bit( stream );
        }
    }

};



#endif //MY_ARITHMATIC_CODER_ARITHMATIC_CODER_H
