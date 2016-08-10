#pragma once

#include <cstdint>

#include <math/math_graphics.h>

namespace gx
{
    struct pinhole_camera
    {

        public:

        //view parameters
        math::float4 position() const
        {
            return m_view_position_ws;
        }

        math::float4 forward() const
        {
            return m_forward_ws;
        }

        math::float4 up() const
        {
            return m_up_ws;
        }

        //perspective parameters
        float				fov() const
        {
            return m_fov;
        }

        float aspect_ratio() const
        {
            return m_aspect_ratio;
        }

        float get_near() const
        {
            return m_near;
        }

        float get_far() const
        {
            return m_far;
        }

        //view parameters
        void set_view_position(math::float4 position_ws)
        {
            m_view_position_ws = position_ws;
        }

        void set_forward(math::float4 view_direction_ws)
        {
            m_forward_ws = view_direction_ws;
        }

        void set_up(math::float4 up_ws)
        {
            m_up_ws = up_ws;
        }

        //perspective parameters
        void set_fov(float	fov)
        {
            m_fov = fov;
        }

        void set_aspect_ratio(float	aspect_ratio)
        {
            m_aspect_ratio = aspect_ratio;
        }

        void set_near(float	value)
        {
            m_near = value;
        }

        void set_far(float	value)
        {
            m_far = value;
        }

    public:

        //view parameters
        math::float4 m_view_position_ws = math::set(0.0f, 0.0f, 0.0f, 1.0f);
        math::float4 m_forward_ws       = math::set(0.0f, 0.0f, 1.0f, 0.0f);;
        math::float4 m_up_ws            = math::set(0.0f, 1.0f, 0.0f, 0.0f);

        //perspective parameters
        float   m_fov          = 3.1415f / 4.0f;
        float   m_aspect_ratio = 16.0f / 9.0f;
        float   m_near         = 0.1f;      //meters
        float   m_far          = 100.0f;	//meters;

    };

    inline math::float4x4 view_matrix(const pinhole_camera * camera)
    {
        using namespace math;
        return view(camera->position(), normalize3(camera->forward()), normalize3(camera->up()));
    }

    inline math::float4x4 view_matrix(const pinhole_camera & camera)
    {
        using namespace math;
        return view(camera.position(), normalize3(camera.forward()), normalize3(camera.up()));
    }

    inline math::float4x4 inverse_view_matrix(const pinhole_camera * camera)
    {
        using namespace math;
        return inverse_view(camera->position(), normalize3(camera->forward()), normalize3(camera->up()));
    }

    inline math::float4x4 inverse_view_matrix(const pinhole_camera & camera)
    {
        using namespace math;
        return inverse_view(camera.position(), normalize3(camera.forward()), normalize3(camera.up()));
    }

    inline math::float4x4 perspective_matrix(const pinhole_camera * camera)
    {
        return math::perspective_fov_lh(camera->fov(), camera->aspect_ratio(), camera->get_near(), camera->get_far());
    }

    inline math::float4x4 perspective_matrix(const pinhole_camera & camera)
    {
        return math::perspective_fov_lh(camera.fov(), camera.aspect_ratio(), camera.get_near(), camera.get_far());
    }

    inline math::float4x4 inverse_perspective_matrix(const pinhole_camera * camera)
    {
        return math::inverse_perspective_fov_lh(camera->fov(), camera->aspect_ratio(), camera->get_near(), camera->get_far());
    }

    inline math::float4x4 inverse_perspective_matrix(const pinhole_camera& camera)
    {
        return math::inverse_perspective_fov_lh(camera.fov(), camera.aspect_ratio(), camera.get_near(), camera.get_far());
    }

    inline math::float4x4 vp_matrix(const pinhole_camera * camera)
    {
        return math::mul(view_matrix(camera), perspective_matrix(camera));
    }

    inline math::float4x4 vp_matrix(const pinhole_camera & camera)
    {
        return math::mul(view_matrix(camera), perspective_matrix(camera));
    }

    inline math::float4x4 inverse_vp_matrix(const pinhole_camera * camera)
    {
        return math::mul(inverse_perspective_matrix(camera), inverse_view_matrix(camera));
    }

    inline math::float4x4 inverse_vp_matrix(const pinhole_camera & camera)
    {
        return math::mul(inverse_perspective_matrix(camera), inverse_view_matrix(camera));
    }

    inline void rotate(pinhole_camera* camera, math::float4 quaternion)
    {
        auto up = camera->up();
        auto direction = camera->forward();

        camera->set_up(math::rotate_vector3(up, quaternion));
        camera->set_forward(math::rotate_vector3(direction, quaternion));
    }

    inline void rotate(pinhole_camera & camera, math::float4 quaternion)
    {
        auto up = camera.up();
        auto direction = camera.forward();

        camera.set_up(math::rotate_vector3(up, quaternion));
        camera.set_forward(math::rotate_vector3(direction, quaternion));
    }
}


