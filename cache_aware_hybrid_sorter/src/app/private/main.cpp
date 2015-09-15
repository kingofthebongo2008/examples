#include "precompiled.h"
#include <cstdint>
#include <limits>
#include <assert.h>

namespace cahs
{
    namespace loser_tree
    {
        static const uint32_t stream_count = 4;

        struct loser_tree_node
        {
            float    m_key;     // compare nodes
            uint32_t m_stream;  // stream where we come from
        };

        typedef uint32_t node_index;

        inline node_index root()
        {
            return 1;
        }

        inline node_index parent(node_index index)
        {
            return index >> 1;
        }

        inline node_index left_child(node_index index)
        {
            return index << 1;
        }

        inline node_index right_child(node_index index)
        {
            return left_child(index) + 1;
        }

        class loser_tree
        {
            public:
            loser_tree(  )
            {
                
            }

            void merge( float* input[stream_count], const size_t input_lengths[stream_count], float* output )
            {
                initialize_tree(input, input_lengths);

                //first round of tournament
                auto winner = get_winner(root());

                auto element_count = stream_count * input_lengths[0];
                auto ostream = output;

                //cycle through all streams and pull elements
                for (auto i = 0U; i < element_count; ++i)
                {
                    auto key = m_nodes[winner].m_key;
                    auto stream = m_nodes[winner].m_stream - stream_count;

                    *ostream++ = key;

                    auto new_key = *input[stream]++;
                    winner = get_new_winner(winner, new_key);
                }

                restore_stream_markers(input, input_lengths);
            }

            private:
            alignas(64) loser_tree_node m_nodes[ stream_count * 2 ];    // 1 + 1 + 2 + 4 ( top, nodes, nodes, leaves )
            float                       m_save [ stream_count];          // 

            const loser_tree_node*  leaves() const
            {
                return &m_nodes[stream_count];
            }

            loser_tree_node*    leaves()
            {
                return &m_nodes[stream_count];
            }

            void initialize_tree( float* streams[stream_count], const size_t input_lengths[ stream_count] )
            {
                auto leaves = this->leaves();
                for (auto i = 0; i < stream_count; ++i)
                {
                    leaves[i].m_key = *streams[i];
                    leaves[i].m_stream = i + stream_count;      // slot in the tree + stream
                }

                initialize_end_of_stream_markers(streams, input_lengths);
            }

            void initialize_end_of_stream_markers(float * streams[stream_count], const size_t input_lengths[stream_count])
            {
                const float  eof_of_stream_marker = std::numeric_limits<float>::infinity();

                for (auto i = 0; i < stream_count; ++i)
                {
                    auto stream_length = input_lengths[i];
                    m_save[i]                   = streams[i][stream_length]; //save last values. case 1: streams can be from separate memory (last element then is the maximum or garbage). case 2: streams can be in the memory sequentially. then stream[last]=stream1[0], but they are moved in the tree already
                    streams[i][stream_length]   = eof_of_stream_marker;      //memory for the last stream should have 1 element more allocated. this is requirement, saves 1 copy
                    streams[i]++;                                            // make the next elements to get ready to go into the tree. the first ones are already there
                }
            }

            void restore_stream_markers(float* streams[stream_count], const size_t input_lengths[stream_count])
            {
                for (auto i = 0; i < stream_count; ++i)
                {
                    auto stream_length = input_lengths[i];
                    streams[i][stream_length] = m_save[i];
                }
            }

            node_index get_winner(node_index root)
            {
                if (root >= stream_count)
                {
                    return root;    //leaf? return it as a winner
                }
                else
                {
                    auto left  = get_winner( left_child(root));
                    auto right = get_winner( right_child(root));

                    auto left_data = m_nodes[left];
                    auto right_data = m_nodes[right];

                    if ( left_data.m_key <= right_data.m_key )
                    {
                        m_nodes[root] = right_data; //store loser
                        return left;                //return winner
                    }
                    else
                    {
                        m_nodes[root] = left_data;  //store loser
                        return right;               //return winner
                    }
                }
            }

            node_index get_new_winner( node_index winner, float new_key )
            {
                m_nodes[winner].m_key = new_key;
                assert(m_nodes[winner].m_stream == winner);

                auto loser = parent(winner);

                while ( loser != 0 )
                {
                    auto key = m_nodes[loser].m_key;

                    //new key is losing to the old one
                    if (new_key > key) 
                    {
                        //swap loser and winner and move up the tree

                        new_key = key;
                        auto new_winner = m_nodes[loser].m_stream;
                        m_nodes[loser] = m_nodes[winner];

                        winner = new_winner;
                    }

                    loser = parent(loser);
                }

                return winner;
            }
        };
    }
}

int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    float  stream_0[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f };

    float* streams[4] = { stream_0, stream_0 + 2,stream_0 + 4, stream_0 + 6 };
    size_t stream_lengths[4] = { 2, 2, 2, 2 };
    float  output[  sizeof( stream_0 ) / sizeof(stream_0[0]) ];

    cahs::loser_tree::loser_tree t;

    t.merge(streams, stream_lengths, &output[0]);
    


    return 0;
}




