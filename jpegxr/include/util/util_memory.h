#ifndef __UTILITY_MEMORY_H__
#define __UTILITY_MEMORY_H__

#include <memory>
#include <type_traits>

namespace std
{
/*
	template<typename T, typename... Args> unique_ptr<T> make_unique( Args&&... args )
	{
		return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
	}
*/

	template<typename T, typename Arg1> unique_ptr<T> make_unique( Arg1&& args )
	{
		return std::unique_ptr<T>( new T( std::forward<Arg1>(args) ) );
	}
}

#endif