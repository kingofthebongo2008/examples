//-----------------------------------------------------------------------------
// File: Framework\Util\Queue.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _QUEUE_H_
#define _QUEUE_H_

template <class ARG_TYPE>
struct QueueNode
{
	ARG_TYPE object;

	QueueNode <ARG_TYPE> *prev;
	QueueNode <ARG_TYPE> *next;
};

template <class ARG_TYPE>
class Queue
{
public:
	Queue()
	{
		m_count = 0;
		m_first = m_last = m_curr = NULL;
	}

	~Queue()
	{
		Clear();
	}

	unsigned int GetCount() const { return m_count; }

	void AddFirst(const ARG_TYPE object)
	{
		QueueNode <ARG_TYPE> *node = new QueueNode <ARG_TYPE>;
		node->object = object;
		InsertNodeFirst(node);
		m_count++;
	}

	void AddLast(const ARG_TYPE object)
	{
		QueueNode <ARG_TYPE> *node = new QueueNode <ARG_TYPE>;
		node->object = object;
		InsertNodeLast(node);
		m_count++;
	}

	bool RemoveCurrent(bool moveForward = true)
	{
		if (m_curr != NULL){
			QueueNode <ARG_TYPE> *newCurr = (moveForward? m_curr->next : m_curr->prev);
			ReleaseNode(m_curr);
			delete m_curr;
			m_curr = newCurr;
			m_count--;
		}
		return (m_curr != NULL);
	}

	bool GoToFirst(){ return (m_curr = m_first     ) != NULL; }
	bool GoToLast (){ return (m_curr = m_last      ) != NULL; }
	bool GoToPrev (){ return (m_curr = m_curr->prev) != NULL; }
	bool GoToNext (){ return (m_curr = m_curr->next) != NULL; }
	bool GoToObject(const ARG_TYPE object)
	{
		m_curr = m_first;
		while (m_curr != NULL && object != m_curr->object)
		{
			m_curr = m_curr->next;
		}
		return (m_curr != NULL);
	}

	ARG_TYPE GetCurrent() const { return curr->object; }

	void Clear()
	{
		while (m_first)
		{
			m_curr = m_first;
			m_first = m_first->next;
			delete m_curr;
		}
		m_last = m_curr = NULL;
		m_count = 0;
	}

	void MoveCurrentToTop()
	{
		if (m_curr != NULL)
		{
			ReleaseNode(m_curr);
			InsertNodeFirst(m_curr);
		}
	}

protected:
	void InsertNodeFirst(QueueNode <ARG_TYPE> *node)
	{
		if (m_first != NULL)
		{
			m_first->prev = node;
		}
		else
		{
			m_last = node;
		}
		node->next = m_first;
		node->prev = NULL;

		m_first = node;
	}

	void InsertNodeLast(QueueNode <ARG_TYPE> *node)
	{
		if (m_last != NULL)
		{
			m_last->next = node;
		}
		else
		{
			m_first = node;
		}
		node->next = NULL;
		node->prev = m_last;

		m_last = node;
	}

	void ReleaseNode(const QueueNode <ARG_TYPE> *node)
	{
		if (node->prev == NULL)
		{
			m_first = node->next;
		}
		else
		{
			node->prev->next = node->next;
		}

		if (node->next == NULL)
		{
			m_last = node->prev;
		}
		else
		{
			node->next->prev = node->prev;
		}
	}

	unsigned int m_count;
	QueueNode <ARG_TYPE> *m_first, *m_last, *m_curr;
};

#endif // _QUEUE_H_
