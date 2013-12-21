#include "img_loader.h"

#include <Windows.h>

#pragma warning(push)
#pragma warning(disable : 4005)
#include <wincodec.h>
#pragma warning(pop)


#include <os/windows/com_pointers.h>
#include <os/windows/com_ptr.h>
#include <os/windows/com_error.h>


namespace example
{
    static inline os::windows::com_ptr<IWICImagingFactory> create_wic_factory()
    {
        using namespace os::windows;

        com_ptr<IWICImagingFactory> factory;
                
        os::windows::throw_if_failed<com_exception> (
        CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        __uuidof(IWICImagingFactory),
        reinterpret_cast<void**> (&factory)
        ) );

        return factory;

    }

    std::unique_ptr< image > create_image ( const wchar_t* image_file_path )
    {
        using namespace os::windows;

        auto factory = create_wic_factory();

        com_ptr<IWICBitmapDecoder> decoder;
        throw_if_failed<com_exception> (  factory->CreateDecoderFromFilename
            (
                image_file_path,
                nullptr,
                GENERIC_READ,
                WICDecodeMetadataCacheOnDemand,
                &decoder
            ) );

        IWICBitmapFrameDecode* frame;

        throw_if_failed<com_exception> (  
        (
            decoder->GetFrame(0, &frame)
        ));

        IWICFormatConverter* bitmap;

        throw_if_failed<com_exception> (  
        (
            factory->CreateFormatConverter(&bitmap)
        ));

        throw_if_failed<com_exception> (  
        (
            bitmap->Initialize( frame, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0f, WICBitmapPaletteTypeCustom)
        ));

        uint32_t width;
        uint32_t height;

        throw_if_failed<com_exception> (  
        (
            frame->GetSize( &width, &height )
        ));


        size_t cuda_row_pitch = (width * 24 + 7) / 8; //3 * sizeof(uint8_t); // align 8;

        // Allocate temporary memory for image
        auto row_pitch = cuda_row_pitch;
        auto image_size = row_pitch * height;

        cuda::memory_buffer p( cuda::allocate<void*>( image_size ) );

        struct __declspec(align(16)) rgb
        {
            uint8_t c[3];
        };

        std::unique_ptr<uint8_t[]> temp( new uint8_t[ image_size + 32 ] );

        throw_if_failed<com_exception> (  
        (
            frame->CopyPixels( 0, static_cast<uint32_t> ( row_pitch ) , static_cast<uint32_t>  ( image_size ) , temp.get() )
        ));

        // element access into this image looks like this
        auto row = 15;
        auto col = 15 * 3;
        rgb* pElement = (rgb*) &temp[ row * cuda_row_pitch + col];        //((rgb*)temp.get()  + row * cuda_row_pitch) + col;
        rgb* pElement1 = pElement+1;

        // Copy input vectors from host memory to GPU buffers.
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( p, temp.get(), image_size, cudaMemcpyHostToDevice) );

        return std::unique_ptr<image> ( new image ( image::format_24bpp_rgb, static_cast<uint32_t> ( cuda_row_pitch ), width, height, std::move( p )  ) );
    }
}

