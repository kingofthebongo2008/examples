//-----------------------------------------------------------------------------
// File: Framework\Util\Array.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <stdlib.h>

template <class ARG_TYPE>
class Array
{
public:
	Array()
	{
		m_count = m_capacity = 0;
		m_list = NULL;
	}
	
	~Array()
	{
		free(m_list);
	}

	ARG_TYPE &operator [] (const unsigned int index) const
	{
		return m_list[index];
	}

	ARG_TYPE *GetArray() const { return m_list; }
	unsigned int GetCount() const { return m_count; }
	unsigned int GetCapacity() const { return m_capacity; }

	unsigned int Add(const ARG_TYPE object)
	{
		if (m_count >= m_capacity)
		{
			if (m_capacity) m_capacity += m_capacity; else m_capacity = 8;
			m_list = (ARG_TYPE *) realloc(m_list, m_capacity * sizeof(ARG_TYPE));
		}
		m_list[m_count] = object;
		return m_count++;
	}

	void FastRemove(const unsigned int index)
	{
		m_count--;
		m_list[index] = m_list[m_count];
	}

	void OrderedRemove(const unsigned int index)
	{
		m_count--;
		memmove(m_list + index, m_list + index + 1, (m_count - index) * sizeof(ARG_TYPE));
	}

	void SetCount(const unsigned int newCount)
	{
		m_capacity = m_count = newCount;
		m_list = (ARG_TYPE *) realloc(m_list, m_capacity * sizeof(ARG_TYPE));		
	}

	void Clear(){ m_count = 0; }
	void Reset()
	{
		m_count = m_capacity = 0;
		free(m_list);
		m_list = NULL;
	}

protected:
	unsigned int m_capacity;
	unsigned int m_count;
	ARG_TYPE *m_list;
};

#endif // _ARRAY_H_
