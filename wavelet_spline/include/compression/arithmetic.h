#ifndef __compression_arithmetic_h__
#define __compression_arithmetic_h__

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

namespace compression
{
    namespace arithmetic
    {
        static const uint32_t   code_value_bits = 16;
    
        typedef int32_t         code_value;

        static const code_value top_value =  ( ( static_cast<code_value>(1) ) << code_value_bits ) - 1;

        static const code_value first_quarter = ( top_value / 4 + 1);
        static const code_value half          = 2 * first_quarter;
        static const code_value third_quarter = 3 * first_quarter;

        static const uint32_t no_of_chars   = 256;
        static const int32_t  eof_symbol    = no_of_chars + 1;
        static const uint32_t no_of_symbols = no_of_chars + 1;
        static const uint32_t max_frequency = 16383;

        class exception : public std::exception
        {

        };

        inline void raise_error()
        {
            throw exception();
        }

        namespace statistics
        {
            struct context
            {
                int32_t     m_cum_frequency[ no_of_symbols + 1];

                int32_t     m_char_to_index[ no_of_chars ];
                int32_t     m_index_to_char[ no_of_symbols ];

                int32_t     m_frequency[ no_of_symbols + 1];

                context() 
                {
                    //setup tables that translate between symbol indexes and chars
                    for ( uint32_t i = 0; i < no_of_chars; ++i)
                    {
                        m_char_to_index[i] = i+1;
                        m_index_to_char[i+1] = i ;
                    }
                }
            };

            inline context intialize_frequencies( context* s,  const int32_t frequency [ no_of_symbols + 1] )
            {
                s->m_cum_frequency[ no_of_symbols ] = 0;

                //setup cumulative frequency counts
                for (int32_t i = no_of_symbols; i > 0; --i)
                {
                    s->m_cum_frequency[i-1] = s->m_cum_frequency[i] + frequency[i];
                }

                if (s->m_cum_frequency[0] > max_frequency )
                {
                    raise_error();
                }

                std::copy ( &frequency[0], &frequency[0] + no_of_symbols+ 1, std::begin ( s->m_frequency ) );

                return *s;
            }

            inline context create_context( const uint8_t* begin, const uint8_t* end )
            {
                context s;

                //analyze initial frequency of symbols
                int32_t    frequency[ no_of_symbols + 1];
                int32_t    symbol_count = 0;
                std::fill( std::begin(frequency), std::end(frequency), 0 );

                //setup eof symbol frequency
                frequency[ no_of_symbols ] = 1;
                //frequency[ 0 ] = 1;

                std::for_each( begin, end, [&]( const uint8_t v ) -> void
                {
                    auto symbol = s.m_char_to_index[v];
                    frequency[symbol] +=1;
                });

                int freq[ no_of_symbols + 1] = 
                { 
              
                };

                return intialize_frequencies(&s, frequency);
            }

            inline context create_context( const int32_t frequency [no_of_symbols + 1] )
            {
                context s;
                return intialize_frequencies( &s, frequency );
            }
        }

        namespace encoder
        {
            struct context
            {
                statistics::context m_stat;

                code_value          m_low;
                code_value          m_high;

                uint32_t            m_bits_to_follow;

                context( const statistics::context& stat ) : 
                m_stat(stat)
                , m_low(0)
                , m_high(top_value)
                , m_bits_to_follow(0)
                {

                }

                template <typename stream> void bit_plus_follow( uint32_t bit, stream& s )
                {
                    s.output_bit( bit );
                    while (m_bits_to_follow > 0)
                    {
                        s.output_bit(!bit);
                        m_bits_to_follow--;
                    }
                }

                template <typename stream> void encode_symbol( int32_t symbol, stream& s)
                {
                    auto range = (m_high - m_low )  + 1;
                    auto freq_0 = m_stat.m_cum_frequency  [ 0 ];

                    auto freq_symbol_1  = m_stat.m_cum_frequency[ symbol - 1] ;
                    auto freq_symbol    = m_stat.m_cum_frequency[ symbol ] ;

                    m_high  = m_low +  ( range *  freq_symbol_1  ) / freq_0  - 1;
                    m_low   = m_low +  ( range *  freq_symbol    ) / freq_0;

                    for (;;)
                    {
                        if (m_high < half )
                        {
                            bit_plus_follow(0, s);
                        }
                        else if (m_low >= half )
                        {
                            bit_plus_follow(1, s );
                            m_low -= half;
                            m_high -= half;
                        }
                        else if ( m_low >= first_quarter && m_high < third_quarter)
                        {
                            m_bits_to_follow +=1;
                            m_low -= first_quarter;
                            m_high -= first_quarter;
                        }
                        else
                        {
                            break;
                        }

                        m_low   = 2 * m_low;
                        m_high  = 2 * m_high + 1;
                    }
                }

                void start_encoding()
                {
                    m_low = 0;
                    m_high = top_value;
                    m_bits_to_follow = 0;
                }

                template <typename stream> void done_encoding( stream& s)
                {
                    m_bits_to_follow +=1;

                    if ( m_low  < first_quarter )
                    {
                        bit_plus_follow ( 0, s );
                    }
                    else
                    {
                        bit_plus_follow ( 1, s );
                    }
                }
            };

            inline context create_context ( const uint8_t* begin, const uint8_t* end )
            {
                return context ( statistics::create_context( begin, end ) );
            }

            template < typename stream >
            statistics::context encode ( const uint8_t* begin, const uint8_t* end, stream& s)
            {
                context c = create_context( begin, end );

                s.start_outputing_bits();
                c.start_encoding();

                std::for_each( begin, end, [&]( const uint8_t v ) -> void
                {
                    auto symbol = c.m_stat.m_char_to_index[v];
                    c.encode_symbol( symbol, s );
                });

                c.encode_symbol ( eof_symbol, s);
                c.done_encoding(s);
                s.done_outputing_bits();

                return c.m_stat;
            }
        
        }

        namespace decoder
        {
            struct context
            {
                statistics::context m_stat;

                code_value          m_low;
                code_value          m_high;
                code_value          m_value;


                context( const statistics::context& stat ) : 
                m_stat(stat)
                , m_low(0)
                , m_high(top_value)
                {

                }

                template <typename stream>
                void start_decoding( stream& s)
                {
                    m_value = 0;

                    //input bits to fill the codevalue
                    for ( int32_t i = 1; i <= code_value_bits; ++i )
                    {
                        m_value = 2 * m_value + s.input_bit();
                    }

                    //full code range
                    m_low = 0;
                    m_high = top_value;
                }

                template <typename stream>
                int32_t decode_symbol ( stream& s )
                {
                    int32_t range;
                    int32_t cum;

                    int32_t symbol;

                    int32_t freq_0 = m_stat.m_cum_frequency[0];

                    range = ( m_high - m_low ) + 1;
                    cum   = ( (  ( m_value - m_low ) + 1 )  * freq_0 - 1 ) / range; 

                    for ( symbol = 1 ; m_stat.m_cum_frequency[ symbol ] > cum ; ++symbol )
                    {

                    }

                    m_high  = m_low +  ( range * m_stat.m_cum_frequency[ symbol - 1 ] ) / freq_0 - 1;
                    m_low   = m_low +  ( range * m_stat.m_cum_frequency[ symbol     ] ) / freq_0;

                    for ( ;; )
                    {
                        if ( m_high < half )
                        {

                        }
                        else if ( m_low >= half )
                        {
                            m_value -= half;
                            m_low   -= half;
                            m_high  -= half;
                        }
                        else if ( m_low >= first_quarter && m_high <third_quarter)
                        {
                            m_value -= first_quarter;
                            m_low   -= first_quarter;
                            m_high  -= first_quarter;
                        }
                        else
                        {
                            break;
                        }

                        m_low   = 2 * m_low;
                        m_high  = 2 * m_high + 1;
                        m_value = 2 * m_value + s.input_bit();
                    }

                    return symbol;
                }
            };

            template < typename stream, typename iterator >
            void decode ( context& c, stream& s, iterator output )
            {
                s.start_inputing_bits();
                c.start_decoding(s);

                for ( ;; )
                {
                    int32_t ch;
                    int32_t symbol;

                    symbol = c.decode_symbol( s );
                    if ( symbol == eof_symbol )
                    {
                        break;
                    }

                    ch = c.m_stat.m_index_to_char[ symbol ];

                    *output++ = ch;
                }
            }
        }

        template < typename iterator >
        class output_bit_stream
        {
            private:

            int32_t     m_buffer;       //bits buffered for output
            int32_t     m_bits_to_go;   //free bits in the buffer

            iterator    m_output;

            public:

            output_bit_stream( iterator& output ) : m_output(output)
            {
                reset();
            }

            output_bit_stream( iterator&& output ) : m_output( std::move(output) )
            {
                reset();
            }

            void reset()
            {
                m_buffer = 0;
                m_bits_to_go = 8;
            }

            void start_outputing_bits()
            {
                reset();
            }

            void done_outputing_bits()
            {
                auto out = static_cast<uint8_t>(m_buffer >> m_bits_to_go );
                *m_output++ = out;
            }

            void output_bit( int32_t bit )
            {
                //put bit on top of buffer
                m_buffer >>=1;

                if (bit)
                {
                    m_buffer |=0x80;
                }

                m_bits_to_go -=1;

                if (m_bits_to_go == 0)
                {
                    auto out = static_cast<uint8_t>(m_buffer);
                    *m_output++=out;
                    m_bits_to_go = 8;
                }
            }
        };

        template <typename iterator>
        class input_bit_stream
        {
            private:

            int32_t m_buffer;       //bits waiting to be input
            int32_t m_bits_to_go;   //free bits in the buffer
            int32_t m_garbage_bits; //number of bits past eof

            std::vector<uint8_t>   m_input;
            uint32_t               m_input_pointer;
            static const int32_t   eof = static_cast<int32_t> (-1);

            iterator               m_begin;
            iterator               m_end;
            iterator               m_pointer;

            public:
        
            input_bit_stream( iterator begin, iterator end ) :
            m_buffer(0)
            , m_bits_to_go(0)
            , m_garbage_bits(0)
            , m_begin(begin)
            , m_end(end)
            , m_pointer(begin)
            {
            }

            void reset()
            {
                m_buffer = 0;
                m_bits_to_go = 0;
                m_garbage_bits = 0;
                m_input_pointer = 0;
            }

            void start_inputing_bits()
            {
                reset();
            }

            int32_t get_byte()
            {
                if ( m_pointer == m_end  )
                {
                    return eof;
                }
                else
                {
                    return static_cast<int32_t> ( *m_pointer++ );
                }
            }

            inline int32_t input_bit()
            {
                int32_t t;

                if ( m_bits_to_go == 0 )
                {
                    m_buffer = get_byte();

                    if (m_buffer == eof )
                    {
                        m_garbage_bits +=1;

                        if (m_garbage_bits > code_value_bits - 2 )
                        {
                            raise_error();
                        }
                    }

                    m_bits_to_go = 8;
                }

                t = m_buffer & 0x1;

                m_buffer >>= 1;
                m_bits_to_go -=1;
                return t;
            }
        };
        namespace helpers
        {
            struct encoded_result
            {
                std::vector<uint8_t> m_result;
                int32_t              m_frequency_table[no_of_symbols + 1];
            };

            inline encoded_result encode( const uint8_t* begin, const uint8_t* end )
            {
                encoded_result r;

                typedef std::back_insert_iterator < std::vector< uint8_t > > back_iterator;

                arithmetic::output_bit_stream< back_iterator  >  s( std::back_inserter( r.m_result ) );

                auto context = arithmetic::encoder::encode( begin, end, s );

                std::copy(std::begin(context.m_frequency), std::end(context.m_frequency), std::begin( r.m_frequency_table ) );

                return r;
            }

            inline std::vector<uint8_t> decode ( const encoded_result& encoded )
            {
                std::vector<uint8_t> r;

                arithmetic::input_bit_stream< std::vector<uint8_t>::const_iterator >  i( encoded.m_result.begin(), encoded.m_result.end()  );
  
                arithmetic::decoder::context decode_context ( arithmetic::statistics::create_context( encoded.m_frequency_table ) ) ;

                arithmetic::decoder::decode( decode_context, i, std::back_inserter( r ) );

                return r;
            }

            inline void example ( )
            {
                uint8_t message[] = { 'a','a','a', 'a', 'a', ' ', 'a', 'a', 'a', 'a' };

                auto r1 = arithmetic::helpers::encode( std::begin( message ), std::end( message ) );

                auto r2 = arithmetic::helpers::decode ( r1 );
            }
        }
    }
}



#endif
