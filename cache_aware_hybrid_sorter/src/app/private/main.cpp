#include "precompiled.h"
#include <cstdint>
#include <limits>

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

            void merge(float* input[stream_count], const size_t input_lengths[stream_count], float* output )
            {
                initialize_tree(input, input_lengths);

                auto winner = get_winner(root());

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
                    m_save[i]                   = streams[i][stream_length];
                    streams[i][stream_length]   = eof_of_stream_marker;
                    streams[i]++;               // make the next elements to get ready to go into the tree. the first ones are already there
                }
            }

            void restore_stream_markers(float* streams[stream_count], const size_t input_lengths[stream_count])
            {
                for (auto i = 0; i < stream_count; ++i)
                {
                    auto stream_length = input_lengths[stream_count];
                    streams[i][stream_length] = m_save[i];
                }
            }

            node_index get_winner(node_index root)
            {
                if (root >= stream_count)
                {
                    return root;
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
        };
    }
}

int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    float  stream_0[3] = { 0.0f, 2.0f, 1.0f };
    float  stream_1[3] = { 0.0f, 2.0f, 1.0f };
    float  stream_2[3] = { 0.0f, 2.0f, 1.0f };
    float  stream_3[3] = { 0.0f, 2.0f, 1.0f };

    float* streams[4] = { stream_0, stream_1,stream_2, stream_3 };
    size_t stream_lengths[4] = { 2, 2, 2, 2 };
    float  output[16];

    cahs::loser_tree::loser_tree t;

    t.merge(streams, stream_lengths, &output[0]);
    


    return 0;
}




