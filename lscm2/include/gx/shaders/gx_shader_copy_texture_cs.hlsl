StructuredBuffer<uint>   source;
RWStructuredBuffer<uint> destination;

[numthreads(1, 1, 1)]
void main( uint3 dispatch_thread_id : SV_DispatchThreadID ) 
{
    uint s  = source[dispatch_thread_id.x];
    destination[dispatch_thread_id.x] = ( 0x00FF00FF);
}