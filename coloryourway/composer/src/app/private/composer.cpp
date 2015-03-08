#include "precompiled.h"

#include <assert.h>
#include <fstream>
#include <string>
#include <list>
#include <vector>

#include <os/windows/com_initializer.h>

#include "composer_application.h"
#include "composer_renderable.h"

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
                *address(m, sample_classes, i, i) = r[i];
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
        }

        struct sample
        {
            float       m_x;        //x coordinate
            float       m_y;        //y coordinate
            uint32_t    m_c;        //sample class
        };

        struct multi_class_dart_throwing_context
        {
            std::vector< uint32_t > m_class_sample_count;           //total samples generated so far
            std::vector< uint32_t > m_class_total_sample_count;     //total samples we need to generate

            std::list< sample >     m_final_set_of_samples;
            uint32_t                m_final_set_of_samples_count;
            uint32_t                m_total_sample_count;

            float*                  m_r;                        //r matrix
            uint32_t                m_total_trials;
            uint32_t                m_sample_classes;           //sample classes count
        };

        static sample inline generate_new_sample()
        {
            sample r;

            auto max = float(RAND_MAX);
            
            r.m_x = std::rand() / max;
            r.m_y = std::rand() / max;
            r.m_c = 0;
            return r;
        }

        static inline float distance(const sample& s1, const sample& s2)
        {
            float dx = s1.m_x - s2.m_x;
            float dy = s1.m_y - s2.m_y;

            return sqrtf(dx * dx + dy * dy);
        }

        static inline float fill_rate(const multi_class_dart_throwing_context* ctx, uint32_t c)
        {
            return static_cast<float>( ctx->m_class_sample_count[c] )  / ctx->m_class_total_sample_count[c];
        }

        static bool removable(const multi_class_dart_throwing_context* ctx, std::vector< std::list<sample>::const_iterator >& ns, sample& s, uint32_t sample_classes, float* r)
        {
            for (auto it = std::cbegin(ns); it != std::cend(ns); ++it)
            {
                auto c1 = (*it)->m_c;
                auto c2 = s.m_c;
                float d1 = *address(r, sample_classes, c1, c1);
                float d2 = *address(r, sample_classes, c2, c2);
                if (  d1 >= d2 || fill_rate ( ctx, c1) < fill_rate(ctx, c2) )
                {
                    return false;
                }
            }
            
            return false;
        }

        static uint32_t find_sample_class(multi_class_dart_throwing_context* c)
        {
            auto min_class = 0;
            auto min_fill_rate = fill_rate(c, 0);

            for (auto i = 0U; i < c->m_sample_classes; ++i)
            {
                auto r = fill_rate(c, i);
                if (r < min_fill_rate)
                {
                    min_class = i;
                    min_fill_rate = r;
                }
            }

            return min_class;
        }

        void multi_class_dart_throwing( multi_class_dart_throwing_context* c )
        {
            auto trials = 0U;

            std::vector< std::list<sample>::const_iterator > ns;
            ns.reserve(1000);

            while (trials < c->m_total_trials && c->m_final_set_of_samples_count < c->m_total_sample_count  )
            {
                auto s  = generate_new_sample();
                auto cs = find_sample_class( c ); 
                s.m_c = cs;

                ns.clear();

                for (auto it = std::cbegin(c->m_final_set_of_samples); it != std::end(c->m_final_set_of_samples); ++it )
                {
                    auto c1 = cs;
                    auto c2 = it->m_c;

                    auto d = *address(c->m_r, c->m_sample_classes, c1, c2);
                    if ( distance(s, *it) < d )
                    {
                        ns.push_back(it);
                    }
                }

               
                if (ns.empty())
                {
                    c->m_final_set_of_samples.push_back(s);
                    c->m_final_set_of_samples_count++;
                    c->m_class_sample_count[s.m_c]++;
                }
                else
                {
                    if (removable(c, ns, s, c->m_sample_classes, c->m_r ) )
                    {
                        for (auto it1 = std::begin(ns); it1 != std::end(ns); ++it1)
                        {
                            c->m_final_set_of_samples.erase(*it1);
                        }

                        c->m_final_set_of_samples.push_back(s);
                        c->m_final_set_of_samples_count++;
                        c->m_class_sample_count[s.m_c]++;
                    }
                }
            }
        }


        std::tuple < std::list<sample>, uint32_t, uint32_t > build_samples()
        {
            const auto sample_classes = 5U;
            const auto sample_count = 80U;

            const float r[sample_classes] = { 0.011f, 0.02f, 0.01f, 0.01f, 0.01f };

            uint32_t ni[sample_classes];

            for (auto i = 0U; i < sample_classes; ++i)
            {
                ni[i] = sample_count_class(i, sample_count, sample_classes, r);
            }

            float rm[sample_classes][sample_classes];

            build_r_matrix(&rm[0][0], sample_classes, r);

            multi_class_dart_throwing_context c;

            c.m_r = &rm[0][0];
            c.m_sample_classes = sample_classes;
            c.m_total_trials = 10000;
            c.m_total_sample_count = sample_count;

            c.m_final_set_of_samples_count = 0;
            c.m_class_sample_count.resize(sample_classes);

            c.m_class_total_sample_count.resize(sample_classes);
            std::copy(ni, ni + sample_classes, std::begin(c.m_class_total_sample_count));

            multi_class_dart_throwing(&c);

            return std::make_tuple(std::move(c.m_final_set_of_samples), c.m_final_set_of_samples_count, sample_classes);
        }

        struct sample_render_info
        {
            std::vector< sample > m_samples;
            uint32_t              m_sample_classes;

            sample_render_info() : m_sample_classes(0)
            {

            }

            sample_render_info(const sample_render_info& o) : m_samples(o.m_samples), m_sample_classes(o.m_sample_classes)
            {

            }

            sample_render_info( sample_render_info&& o ) : m_samples( std::move(o.m_samples)), m_sample_classes(std::move(o.m_sample_classes))
            {

            }


            sample_render_info& operator=(const sample_render_info& o)
            {
                m_samples = o.m_samples;
                m_sample_classes = o.m_sample_classes;
                return *this;
            }

            sample_render_info& operator=(sample_render_info&& o)
            {
                m_samples = std::move(o.m_samples);
                m_sample_classes = std::move(o.m_sample_classes);
                return *this;
            }

        };

        class samples_renderable : public renderable
        {
            public:

            explicit samples_renderable(const sample_render_info& samples) : m_samples(samples)
            {

            }

            explicit samples_renderable(sample_render_info && samples) : m_samples(std::move(samples))
            {

            }

            private:

            sample_render_info m_samples;

            void on_draw( render_context& c )
            {
                


            }
        };
    }
}


int32_t APIENTRY _tWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow)
{
    using namespace coloryourway::composer;
    using namespace std;

    auto samples = build_samples();

    vector< sample > v_samples;

    v_samples.resize(get<1>(samples));

    copy(begin(get<0>(samples)), end(get<0>(samples)), begin(v_samples));

    sample_render_info info;
    info.m_sample_classes = get<2>(samples);
    info.m_samples = move(v_samples);

    os::windows::com_initializer com ( os::windows::apartment_threaded) ;
    
    auto app = new sample_application(L"Composer");

    
    auto result = app->run();

    delete app;

    return 0;
}




