#include "precompiled.h"

#include <assert.h>
#include <fstream>
#include <string>

#include "composer_application.h"
#include <os/windows/com_initializer.h>

namespace coloryourway
{
    namespace composer
    {

        static uint32_t sample_count_class(uint32_t sample_class, uint32_t sample_count, uint32_t sample_classes, const float r[])
        {
            auto sum = 0.0f;
            for (auto i = 0U; i < sample_classes; ++i)
            {
                sum += 1.0f / (r[i] * r[i]);
            }

            auto ri = r[sample_class];
            auto inv_ri = 1.0f / (ri * ri);
            return static_cast<uint32_t> (sample_count * inv_ri / sum);
        }


        static inline uint32_t offset(uint32_t sample_classes, uint32_t i, uint32_t j)
        {
            return i * sample_classes + j;
        }

        static inline float* address(float* m, uint32_t sample_classes, uint32_t i, uint32_t j)
        {
            return m + offset(sample_classes, i, j);
        }

        struct priority_group
        {
            priority_group() : m_r(0.0f)
            {

            }

            priority_group(float r, uint32_t index)
            {
                m_r = r;
                m_r_index.push_back(index);
            }

            float                  m_r;
            std::vector< uint32_t> m_r_index;
        };

        static bool find_priority_group(std::vector<priority_group>& groups, float r, uint32_t index)
        {
            for (auto i = 0U; i < groups.size(); ++i)
            {
                auto& g = groups[i];
                if (g.m_r == r)
                {
                    g.m_r_index.push_back(index);
                    return true;
                }
            }

            return false;
        }

        struct sort_priority_group
        {
            inline bool operator()(const priority_group& a, const priority_group& b) const
            {
                return b.m_r < a.m_r;
            }
        };

        static void build_r_matrix(float* m, uint32_t sample_classes, const float r[])
        {
            for (uint32_t i = 0U; i < sample_classes; ++i)
            {
                *address(m, sample_classes, i, i) = 1.0f;
            }

            std::vector < priority_group > groups;
            groups.reserve(sample_classes);

            for (auto i = 0U; i < sample_classes; ++i)
            {
                if (!find_priority_group(groups, r[i], i))
                {
                    groups.push_back(priority_group(r[i], i));
                }
            }

            std::sort(groups.begin(), groups.end(), sort_priority_group());

            std::vector<uint32_t > c;   //the set of classes already processed
            float d = 0.0f;              //the density of the classes already processed


            for (auto k = 0U; k < groups.size(); ++k)
            {
                auto& g = groups[k];

                for (auto i = 0U; i < g.m_r_index.size(); ++i)
                {
                    c.push_back(g.m_r_index[i]);
                    d = d + 1.0f / (g.m_r * g.m_r);
                }

                for (auto g1 = 0U; g1 < g.m_r_index.size(); ++g1)
                {
                    auto i = g.m_r_index[g1];

                    for (auto h = 0U; h < c.size(); ++h)
                    {
                        auto j = c[h];

                        if ( i != j  )
                        {
                            auto v = 1.0f / sqrtf(d);
                            *address(m, sample_classes, i, j) = v;
                            *address(m, sample_classes, j, i) = v;
                        }
                    }
                }
            }


            __debugbreak();
        }

    }
}

int32_t APIENTRY _tWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow)
{
    using namespace coloryourway::composer;

    const auto sample_classes = 5U;
    const auto sample_count = 30U;

    const float r[sample_classes] = { 0.01f, 0.02f, 0.03f, 0.04f, 0.05f };

    uint32_t ni[sample_classes];

    for (auto i = 0U; i < sample_classes; ++i)
    {
        ni[i] = sample_count_class(i, sample_count, sample_classes, r);
    }

    float rm[sample_classes][sample_classes];

    build_r_matrix(&rm[0][0], sample_classes, r);


 
    /*
    os::windows::com_initializer com ( os::windows::apartment_threaded) ;
    using namespace coloryourway::composer;
    auto app = new sample_application(L"Composer");
    
    auto result = app->run();

    delete app;
    */

    return 0;
}




