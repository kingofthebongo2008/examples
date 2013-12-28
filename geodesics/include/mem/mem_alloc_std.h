#ifndef __MEM_ALLOC_STD_H__
#define __MEM_ALLOC_STD_H__

#include <malloc.h>

namespace mem
{
	template<typename T>
	class allocator
	{
	public : 

		typedef T value_type;
		typedef value_type* pointer;
		typedef const value_type* const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

	public : 
		template<typename U>
		struct rebind
		{
			typedef allocator<U> other;
		};

	public : 
		allocator() {}
		~allocator() {}
		allocator(allocator const&) {}
		template<typename U>
		explicit allocator(allocator<U> const&) {}

		
		pointer address(reference r) { return &r; }
		const_pointer address(const_reference r) { return &r; }

		pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0)
		{ 
		  return reinterpret_cast<pointer>( _aligned_malloc(cnt * sizeof (T), 16 ) );
		}

		void deallocate(pointer p, size_type)
		{ 
			_aligned_free(p);
		}

		size_type max_size() const
		{ 
			return std::numeric_limits<size_type>::max() / sizeof(T);
		}

		void construct(pointer p, const T& t) { new(p) T(t); }
		void destroy(pointer p) { if(p) p->~T(); }

		bool operator==(allocator const&) { return true; }
		bool operator!=(allocator const& a) { return !operator==(a); }
	}; 
}


#endif
