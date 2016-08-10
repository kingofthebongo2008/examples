// ucdev_include_parser.cpp : Defines the entry point for the console application.
//
#include "precompiled.h"

#include <fstream>
#include <streambuf>
#include <sstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <experimental/filesystem>
#include "image_map.h"
#include <ppl.h>
#include <ppltasks.h>
#include <math/math_functions.h>

using tile  = uint16_t;
using texel = uint16_t;

inline texel to_texel(tile t)
{
    return t * 128U;
}

inline tile to_tile(texel t)
{
    return t / 128U;
}

struct map
{
    tile m_width    = 128;
    tile m_height   = 128;
};

template <uint32_t n> inline size_t mip_levels()
{
    return math::log2_c<n>::value + 1;
}

inline std::string make_tile_file_name( tile row, tile col )
{
    std::stringstream str;
    str << "tile_"<< std::setfill('0') << std::setw(4) << row << "_" << std::setfill('0') << std::setw(4) << col << ".tga" ;

    return str.str();
}

inline std::string make_tile_file_name( uint32_t mip, tile row, tile col )
{
    std::stringstream str;
    str << "tile_" << std::setfill('0') << std::setw(4) <<  mip << "_"<< std::setfill('0') << std::setw(4) << row << "_" << std::setfill('0') << std::setw(4) << col << ".tga";
    return str.str();
}

void generate_tiles( const map& m, uint32_t mip_level )
{
    auto width  = m.m_width;
    auto height = m.m_height;

    auto w = width  >> mip_level;
    auto h = height >> mip_level;

    concurrency::parallel_for(0, w * h, [w, h, mip_level](auto index)
    {
        auto row = index / h;
        auto col = index % w;
            
        auto file_name = make_tile_file_name(mip_level, row, col );

        map_builder::image::color c[3] = {

            map_builder::image::color::red(),
            map_builder::image::color::green(),
            map_builder::image::color::blue()
        };
        
        auto img = map_builder::make_image(128, 128, c[ mip_level % 3] );
        map_builder::save_image(img.get(), file_name);
    });
}

concurrency::task<void> generate_tiles(const map& m)
{
    return concurrency::create_task([&m] ()
    {
        concurrency::task_group g;

        auto l = mip_levels<128>();

        std::cout << l << std::endl;

        for (auto i = 0; i < mip_levels<128>(); ++i)
        {
            g.run([i, &m] { generate_tiles(m, i);}  );
        }

        g.wait();
    });
}


int32_t main(int32_t argc, const char* argv[])
{
    auto t = generate_tiles(map());

    while ( !t.is_done())
    {
        std::cout << ".";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    

	return 0;
}


