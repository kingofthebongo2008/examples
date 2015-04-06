#include "precompiled.h"

#include <atomic>
#include <chrono>

#include <cstdint>
#include <system_error>

#include "intrusive_ptr.h"

namespace vts
{
    enum class future_errc
    {
        broken_promise,
        future_already_retrieved,
        promise_already_satisfied,
        no_state
    };

    enum class future_status
    {
        ready = 0,
        timeout = 1,
        deferred = 2
    };

    class future_error : public std::logic_error
    {
    public:

        explicit future_error(std::error_code code) : logic_error(""), m_code(code)
        {

        }

        const char* what() const throw() override
        {
            return m_code.message().c_str();
        }

        const std::error_code code() const throw()
        {
            return m_code;
        }

    private:

        std::error_code m_code;
    };
}

namespace std
{
    template <> struct is_error_code_enum<vts::future_errc> : public std::true_type{};
}

namespace vts
{
    namespace details
    {
        template <typename derived>
        class refcount_base
        {
            private:

            typedef refcount_base<derived> this_type;

            friend void intrusive_ptr_add_ref(const derived* pointer)
            {
                const this_type* t = static_cast<const this_type*>(pointer);
                ++t->m_ref_count;
            }

            friend void intrusive_ptr_release(const derived* pointer)
            {
                const this_type* t = static_cast<const this_type*>(pointer);

                if ( --t->m_ref_count == 0 )
                {
                    if ( 0 < sizeof( derived) )
                    {
                        delete pointer;
                    }
                }
            }

            protected:

            refcount_base() : m_ref_count(0) {}
            refcount_base(const refcount_base&) : m_ref_count(0) {}
            refcount_base(const refcount_base&&) : m_ref_count(0) {}
            refcount_base& operator=(const refcount_base&) { return *this; }
            refcount_base&& operator=(const refcount_base&&) { return *this; }
            ~refcount_base() {};

            void swap(refcount_base&) {};
            mutable std::atomic_uint32_t m_ref_count;
        };


        template <typename t>
        class future_shared_state : public refcount_base< future_shared_state<t> >
        {

        public:
            ~future_shared_state()
            {

            }
        };

        class future_error_category : public std::error_category
        {
            const char *name() const throw() override
            {
                return "Future";
            }

            std::string message(std::int32_t error) const override
            {
                switch ( future_errc(error) )
                {
                    case (future_errc::broken_promise) :
                    {
                        return "Future Error: Broken Promise";
                    }

                    case (future_errc::future_already_retrieved) :
                    {
                        return "Future Error: Future Already Retrieved";
                    }

                    case (future_errc::promise_already_satisfied) :
                    {
                        return "Future Error: Promise Already Satisfied";
                    }

                    case (future_errc::no_state) :
                    {
                        return "Future Error: No Associated State";
                    }
                    default:
                    {
                        return std::string();
                    }
                }
            }
        };
    }

    inline std::error_category& future_category() throw()
    {
        static details::future_error_category category;
        return category;
    }

    inline std::error_code make_error_code( future_errc e )
    {
        return std::error_code(static_cast<int>(e), future_category());
    }

    inline std::error_condition make_error_condition(future_errc e)
    {
        return std::error_condition(static_cast<int>(e), future_category());
    }

    namespace details
    {
        inline void throw_future_error(future_errc c)
        {
            throw future_error( make_error_code(c) );
        }
    }

    template <class r> class future
    {
        public:

        future() throw()
        {

        
        }

        future( future&& o )
        {

        }

        ~future()
        {

        }

        future& operator = (future&& o) throw()
        {

        }

        r get()
        {

        }

        bool valid() const throw()
        {
            return false;
        }

        void wait() const
        {

        }

        template <class rep, class period>
        future_status wait_for(const std::chrono::duration<rep, period>& rel_time) const
        {
            return future_status::ready;

        }

        template <class clock, class duration>
        future_status wait_until(const std::chrono::time_point<clock, duration>& abs_time) const
        {
            return future_status::ready;
        }

        private:

        future(const future&) = delete;
        future& operator= (const future&) = delete;

        intrusive_ptr< details::future_shared_state<r> > m_shared_state;
    };
}


int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    using namespace vts;

    intrusive_ptr< details::future_shared_state<int> > f(new details::future_shared_state<int>());
    
    return 0;
}




