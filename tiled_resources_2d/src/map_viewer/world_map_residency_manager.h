#pragma once

#include <list>
#include <iterator>
#include <memory>

namespace app
{
    namespace tile_residency
    {
        static const unsigned int pool_size_in_tiles                = 256;
        static const unsigned int max_simulataneous_file_load_tasks = 10;
        static const unsigned int max_tiles_loaded_per_frame        = 100;
    }

    inline D3D11_TILED_RESOURCE_COORDINATE to_d3d_tile_coords(tile_coordinates s)
    {
        D3D11_TILED_RESOURCE_COORDINATE r = {};

        r.X           = s.m_tile_x;
        r.Y           = s.m_tile_y;
        r.Subresource = s.m_mip;
        return r;
    }

    inline tile_coordinates to_tile_coords( D3D11_TILED_RESOURCE_COORDINATE s )
    {
        tile_coordinates r = {};
        r.m_tile_x = s.X;
        r.m_tile_y = s.Y;
        r.m_mip = s.Subresource;

        return r;
    }

    D3D11_TILE_REGION_SIZE one_region()
    {
        D3D11_TILE_REGION_SIZE r = {};
        r.NumTiles = 1;
        return r;
    }

    struct world_map_residency_manager
    {
        enum tile_state : uint32_t
        {
            seen = 0,
            loading = 1,
            loaded = 2,
            mapped = 3
        };

        struct tracked_tile
        {
            D3D11_TILED_RESOURCE_COORDINATE m_coordinate = {};
            uint64_t                        m_last_seen_frame = 0;

            void*                           m_tile_data;
            size_t                          m_tile_data_size = 0;

            uint32_t                        m_physical_tile_offset = 0;
            tile_state                      m_state = tile_state::seen;
        };

        struct tile_mapping_update_arguments
        {
            std::vector<D3D11_TILED_RESOURCE_COORDINATE> m_coordinates;
            std::vector<UINT>                            m_range_flags;
            std::vector<UINT>                            m_physical_offsets;
            
            // For convenience, the tracked tile mapping is also saved.
            std::list < std::shared_ptr<tracked_tile>>   m_tiles_to_map;
        };

        tiled_texture* m_tiled_texture = nullptr;
        std::unordered_map< tile_coordinates, std::shared_ptr< tracked_tile > > m_tracked_tiles;

        std::unique_ptr< uint8_t[] >                    m_tile_data;
        size_t                                          m_tile_data_size = 0;
        std::atomic<uint32_t>                           m_active_loading_tasks = 0;

        std::list< std::shared_ptr< tracked_tile> >     m_seen_tiles;       //state machine with tiles, all tiles are processed by these list, until they get mapped
        std::list< std::shared_ptr< tracked_tile> >     m_loading_tiles;     //when they are unmapped, they are not tracked anymore
        std::list< std::shared_ptr< tracked_tile> >     m_mapped_tiles;

        static const uint32_t                           m_reserved_tiles     = 1;
        static const uint32_t                           m_default_tile_index = 0; //null tile index

        static bool newer_tiles_first(const std::shared_ptr< tracked_tile > & a, const std::shared_ptr<tracked_tile>& b)
        {
            // Prefer more recently seen tiles.
            if (a->m_last_seen_frame > b->m_last_seen_frame)
            {
                return true;
            }

            if (a->m_last_seen_frame < b->m_last_seen_frame)
            {
                return false;
            }

            // Break ties by loading less detailed tiles first.
            return a->m_coordinate.Subresource > b->m_coordinate.Subresource;
        }

        static bool older_tiles_first(const std::shared_ptr< tracked_tile > & a, const std::shared_ptr<tracked_tile>& b)
        {
            // Prefer more recently seen tiles.
            if (a->m_last_seen_frame < b->m_last_seen_frame)
            {
                return true;
            }

            if (a->m_last_seen_frame > b->m_last_seen_frame)
            {
                return false;
            }

            // Break ties by loading less detailed tiles first.
            return a->m_coordinate.Subresource > b->m_coordinate.Subresource;
        }

        static bool loaded_tiles_first(const std::shared_ptr< tracked_tile > & a, const std::shared_ptr<tracked_tile>& b)
        {
            if (a->m_state == tile_state::loaded && b->m_state == tile_state::loading) return true;
            if (a->m_state == tile_state::loading && b->m_state == tile_state::loaded) return false;

            return newer_tiles_first(a, b);
        }

        static bool is_older(const tracked_tile* a, const tracked_tile* b)
        {
            return a->m_last_seen_frame < b->m_last_seen_frame;
        }

        void process_samples( const std::vector< tile_coordinates >& samples, size_t frame_count)
        {
            for ( auto&& s : samples )
            {
                auto&& tile = m_tracked_tiles.find(to_tile_coords( to_d3d_tile_coords(s)));

                if ( tile == std::end(m_tracked_tiles) )
                {
                    //load this mip and the mips above it up to 7
                    auto mip = std::max(0U, std::min<uint32_t>(7U, s.m_mip));

                    for (auto m = mip; m < 7; ++m)
                    {
                        auto new_tile = std::make_shared< tracked_tile >();

                        D3D11_TILED_RESOURCE_COORDINATE t = {};

                        t.Subresource = m;
                        t.X = uint32_t( s.m_tile_x / exp2f( (float) (m - s.m_mip) ));
                        t.Y = uint32_t( s.m_tile_y / exp2f( (float) (m - s.m_mip) ) );

                        uint32_t test1 = s.m_tile_x / ( 1 << (m - s.m_mip) );
                        uint32_t test2 = s.m_tile_y / ( 1 << (m - s.m_mip));

                        assert(test1 == t.X);
                        assert(test2 == t.Y);
                        

                        new_tile->m_coordinate      = t;//to_d3d_tile_coords(s);

                        new_tile->m_tile_data       = &m_tile_data[0];
                        new_tile->m_tile_data_size  = m_tile_data_size;
                        new_tile->m_last_seen_frame = frame_count;

                        m_tracked_tiles.insert(std::make_pair(to_tile_coords(new_tile->m_coordinate), new_tile));
                        m_seen_tiles.push_back(new_tile);

                    }
                }
                else
                {
                    tile->second->m_last_seen_frame = frame_count;
                }
            }
        }

        void update(ID3D11DeviceContext2* ctx)
        {
            m_seen_tiles.sort(newer_tiles_first);
            m_loading_tiles.sort(loaded_tiles_first);
            m_mapped_tiles.sort(older_tiles_first);

            m_active_loading_tasks = 0;

            //todo: do a cancel ops here
            for (auto i = m_active_loading_tasks.load(); i < tile_residency::max_simulataneous_file_load_tasks && !m_seen_tiles.empty(); ++i)
            {
                auto tile = m_seen_tiles.front();
                m_seen_tiles.pop_front();
                m_loading_tiles.push_back(tile);
                tile->m_state = tile_state::loaded;
            }

            tile_mapping_update_arguments coalesced_mapping_arguments;

            for (auto i = 0; i < tile_residency::max_tiles_loaded_per_frame && !m_loading_tiles.empty(); ++i)
            {
                auto tile = m_loading_tiles.front();

                if (tile->m_state != tile_state::loaded )  //we got to the loading states
                {
                    // This sample's residency management assumes that for a given texcoord,
                    // there will never be a detailed MIP resident where a less detailed one
                    // is NULL-mapped. This is enforced by sort predicates. A side-effect of
                    // this technique is that mapping cannot occur out of order.

                    break;
                }

                m_loading_tiles.pop_front();

                auto mapped_size = m_mapped_tiles.size() + m_reserved_tiles;
                uint32_t physical_tile_offset = static_cast<uint32_t>(mapped_size);

                if ( mapped_size == tile_residency::pool_size_in_tiles)
                {
                    auto tile_to_evict = m_mapped_tiles.front();

                    if (is_older(tile.get(), tile_to_evict.get()))
                    {
                        // If the candidate tile to map is older than the eviction candidate,
                        // skip the mapping and discard it. This can occur if a tile load stalls,
                        // and by the time it is ready it has moved off-screen.

                        m_tracked_tiles.erase(to_tile_coords(tile->m_coordinate));
                        continue;;
                    }

                    m_mapped_tiles.pop_front();

                    physical_tile_offset = tile_to_evict->m_physical_tile_offset; //remember the offset

                    //unmap the evicted tile
                    coalesced_mapping_arguments.m_coordinates.push_back(tile_to_evict->m_coordinate);
                    coalesced_mapping_arguments.m_physical_offsets.push_back(0);
                    coalesced_mapping_arguments.m_range_flags.push_back(D3D11_TILE_RANGE_REUSE_SINGLE_TILE);
                }

                //map the new tile arguments
                coalesced_mapping_arguments.m_coordinates.push_back(tile->m_coordinate);
                coalesced_mapping_arguments.m_physical_offsets.push_back(physical_tile_offset);
                coalesced_mapping_arguments.m_range_flags.push_back(0);
                coalesced_mapping_arguments.m_tiles_to_map.push_back(tile);

                tile->m_state = tile_state::mapped;
                tile->m_physical_tile_offset = physical_tile_offset;
                m_mapped_tiles.push_back(tile);
            }

            auto&& mapping = coalesced_mapping_arguments;
            if (!coalesced_mapping_arguments.m_coordinates.empty())
            {
                std::vector<uint32_t> range_counts(mapping.m_range_flags.size(), 1);
                std::vector<D3D11_TILE_REGION_SIZE> sizes(mapping.m_range_flags.size(), one_region());

                d3d11::helpers::update_tile_mappings( 
                    ctx,
                    m_tiled_texture->m_resource.get(),
                    static_cast<uint32_t>(mapping.m_coordinates.size()),
                    &coalesced_mapping_arguments.m_coordinates[0],
                    &sizes[0],
                    m_tiled_texture->m_tile_pool.get(),
                    static_cast<uint32_t>(mapping.m_range_flags.size()),
                    &coalesced_mapping_arguments.m_range_flags[0],
                    &coalesced_mapping_arguments.m_physical_offsets[0],
                    &range_counts[0],
                    0
                );

                ctx->TiledResourceBarrier(nullptr, m_tiled_texture->m_resource.get());

                //update the tiles data
                for (auto i = 0; i < mapping.m_coordinates.size(); ++i)
                {
                    if (mapping.m_range_flags[i] == 0)
                    {
                        auto region = one_region();
                        auto tile = mapping.m_tiles_to_map.front();

                        ctx->UpdateTiles
                        (
                            m_tiled_texture->m_resource.get(),
                            &mapping.m_coordinates[i],
                            &region,
                            tile->m_tile_data,
                            0
                        );

                        mapping.m_tiles_to_map.pop_front();
                        ctx->TiledResourceBarrier(nullptr, m_tiled_texture->m_resource.get());
                    }
                }
            }
        }
    };

    std::unique_ptr<world_map_residency_manager> make_residency_manager(ID3D11Device* d, tiled_texture* t)
    {
        std::unique_ptr<world_map_residency_manager> r = std::make_unique<world_map_residency_manager>();

        auto size = 128 * 128 * sizeof(uint32_t);
        std::unique_ptr< uint8_t[] > tile_data(new uint8_t[size]);

        std::memset(&tile_data[0], 0xff, size);
        
        r->m_tile_data_size = size;
        r->m_tile_data = std::move(tile_data);

        r->m_tiled_texture = t;

        return r;
    }
        
}

    

