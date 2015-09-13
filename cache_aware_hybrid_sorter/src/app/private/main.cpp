#include "precompiled.h"
#include <cstdint>
#include <limits>

namespace cahs
{
    namespace looser_tree
    {
        static const uint32_t stream_count = 4;

        struct loser_tree_node
        {
            float    m_key;     // compare nodes
            uint32_t m_stream;  // stream where we come from
        };

        class loser_tree
        {
            void merge( float* input[stream_count], const size_t input_lengths[stream_count], float* output )
            {

            }

            private:

            alignas(64) loser_tree_node m_nodes[ stream_count * 2 ];    // 1 + 1 + 2 + 4 ( top, nodes, nodes, leaves )
            float                       m_save[ stream_count];          //!< to restore the locations of the sentinels

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
                    auto stream_length = input_lengths[stream_count];
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
        };
       
        typedef uint32_t node_index;
        inline node_index parent(node_index index)
        {
            return index >> 1;
        }

        inline node_index left_child(node_index index)
        {
            return index << 1;
        }

        inline node_index right_child( node_index index )
        {
            return left_child(index) + 1;
        }
    }
}

int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    float  stream_0[2] = { 0.0f, 2.0f };
    float  stream_1[2] = { 0.0f, 2.0f };
    float  stream_2[2] = { 0.0f, 2.0f };
    float  stream_3[2] = { 0.0f, 2.0f };

    float* streams[4] = { stream_0, stream_1,stream_2, stream_3 };

    streams[0]++;
    streams[1]++;
    streams[2]++;
    streams[3]++;

    return 0;
}




