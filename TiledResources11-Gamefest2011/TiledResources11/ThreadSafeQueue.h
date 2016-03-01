//--------------------------------------------------------------------------------------
// ThreadSafeQueue.h
//
// Templated wrappers that add thread safety to STL queue and priority queue classes.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <queue>

//--------------------------------------------------------------------------------------
// Name: ThreadSafeQueue<T>
// Desc: Implements a thread safe queue that contains objects of type T.
//       The queue only has rear push and front pop exposed.
//--------------------------------------------------------------------------------------
template<class T>
class ThreadSafeQueue
{
protected:
    // The unprotected STL queue:
    std::queue<T> m_Queue;

    // A critical section that protects the queue:
    CRITICAL_SECTION m_CritSec;

public:
    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue constructor
    //--------------------------------------------------------------------------------------
    ThreadSafeQueue()
    {
        InitializeCriticalSection( &m_CritSec );
    }
    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue destructor
    //--------------------------------------------------------------------------------------
    ~ThreadSafeQueue()
    {
        DeleteCriticalSection( &m_CritSec );
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue::SafeTryGet
    // Desc: Attempts to retrieve the item at the front of the queue. If it is successful,
    //       the item is copied into the pItem parameter, along with a TRUE return value.
    //       Otherwise, pItem is unmodified and FALSE is returned.
    //--------------------------------------------------------------------------------------
    BOOL SafeTryGet( T* pItem )
    {
        EnterLock();
        if( m_Queue.empty() )
        {
            ExitLock();
            return FALSE;
        }
        T Item = m_Queue.front();
        m_Queue.pop();
        ExitLock();

        *pItem = Item;

        return TRUE;
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue::SafeAddItem
    // Desc: Adds an item to the back of the queue in a thread safe fashion.
    //--------------------------------------------------------------------------------------
    VOID SafeAddItem( T Item )
    {
        EnterLock();
        m_Queue.push( Item );
        ExitLock();
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue::Size
    // Desc: Returns the number of items in the queue.
    //--------------------------------------------------------------------------------------
    UINT Size() const
    {
        return (UINT)m_Queue.size();
    }

protected:
    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue::EnterLock
    // Desc: Enters the critical section for this queue.
    //--------------------------------------------------------------------------------------
    VOID EnterLock()
    {
        EnterCriticalSection( &m_CritSec );
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeQueue::ExitLock
    // Desc: Exits the critical section for this queue.
    //--------------------------------------------------------------------------------------
    VOID ExitLock()
    {
        LeaveCriticalSection( &m_CritSec );
    }
};

//--------------------------------------------------------------------------------------
// Name: ThreadSafePriorityQueue<T>
// Desc: Implements a thread safe priority queue that contains objects of type T, with
//       UINT priority values.  Lower priority values go to the front of the queue.
//--------------------------------------------------------------------------------------
template<class T>
class ThreadSafePriorityQueue
{
    //--------------------------------------------------------------------------------------
    // Name: QueueEntry
    // Desc: A single entry in the priority queue.
    //--------------------------------------------------------------------------------------
    struct QueueEntry
    {
        UINT Priority;
        T Item;

        //--------------------------------------------------------------------------------------
        // Name: operator<
        // Desc: Sort predicate for queue entries.
        //--------------------------------------------------------------------------------------
        bool operator< ( const QueueEntry& RHS ) const { return Priority > RHS.Priority; }
    };
protected:
    // The unprotected STL priority queue:
    std::priority_queue<QueueEntry> m_Queue;

    // The critical section that protects the priority queue:
    CRITICAL_SECTION m_CritSec;

public:
    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue constructor
    //--------------------------------------------------------------------------------------
    ThreadSafePriorityQueue()
    {
        InitializeCriticalSection( &m_CritSec );
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue destructor
    //--------------------------------------------------------------------------------------
    ~ThreadSafePriorityQueue()
    {
        DeleteCriticalSection( &m_CritSec );
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue::SafeTryGet
    // Desc: Attempts to retrieve the item at the front of the queue. If it is successful,
    //       the item is copied into the pItem parameter, along with a TRUE return value.
    //       Otherwise, pItem is unmodified and FALSE is returned.
    //--------------------------------------------------------------------------------------
    BOOL SafeTryGet( T* pItem )
    {
        EnterLock();
        if( m_Queue.empty() )
        {
            ExitLock();
            return FALSE;
        }
        QueueEntry QE = m_Queue.top();
        m_Queue.pop();
        ExitLock();

        *pItem = QE.Item;

        return TRUE;
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue::SafeAddItem
    // Desc: Adds a new item to the queue in a thread safe manner, using the given priority.
    //--------------------------------------------------------------------------------------
    VOID SafeAddItem( UINT Priority, T Item )
    {
        EnterLock();
        QueueEntry QE;
        QE.Item = Item;
        QE.Priority = Priority;
        m_Queue.push( QE );
        ExitLock();
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue::Size
    // Desc: Returns the number of items in the priority queue.
    //--------------------------------------------------------------------------------------
    UINT Size() const
    {
        return m_Queue.size();
    }

protected:
    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue::EnterLock
    // Desc: Enters the critical section for this queue.
    //--------------------------------------------------------------------------------------
    VOID EnterLock()
    {
        EnterCriticalSection( &m_CritSec );
    }

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafePriorityQueue::ExitLock
    // Desc: Exits the critical section for this queue.
    //--------------------------------------------------------------------------------------
    VOID ExitLock()
    {
        LeaveCriticalSection( &m_CritSec );
    }
};
