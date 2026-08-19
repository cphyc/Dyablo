#pragma once

#include <map>
#include <set>
#include <iostream>

#include <Kokkos_Core.hpp>

namespace dyablo{

namespace MpiBufferPool_Impl{

struct alloc_t
{
  size_t size;
  void* ptr;

  bool operator<( const alloc_t& o ) const
  {
    return std::make_pair(size, ptr) < std::make_pair(o.size, o.ptr);
  }
};

template< class memory_space >
class pool_t
{
private:
  std::map< void*, size_t > used_allocs;
  std::set< alloc_t > free_allocs;

public:
  pool_t() = default;
  ~pool_t()
  {
    clear();
  }

  void clear()
  {
    if( used_allocs.size() != 0 )
    {
      std::cout << "WARNING - ~MpiBufferPool : some allocations are still in use." << std::endl;
    }
    
    for( const alloc_t& a : free_allocs )
    {
      Kokkos::kokkos_free<memory_space>( a.ptr );
    }
    free_allocs.clear();
  }

  void* alloc( size_t req_size )
  {
    alloc_t alloc{};
    if( !this->free_allocs.empty() )
    {
      alloc = *this->free_allocs.rbegin(); // Get the biggest alloc
      this->free_allocs.erase(alloc);
    }

    if( alloc.size <= req_size )
    {
      if( alloc.ptr )
        Kokkos::kokkos_free<memory_space>( alloc.ptr );
      
      //std::cout << "MPI buffers realloc " << alloc.size << " -> " << req_size * 1.3 << std::endl;

      alloc.size = req_size * 1.3;
      alloc.ptr = Kokkos::kokkos_malloc<memory_space>( "MPI Buffer", alloc.size );
    }

    this->used_allocs.insert( std::make_pair( alloc.ptr, alloc.size ) );

    return alloc.ptr;
  }

  void free( void* ptr )
  {   
    alloc_t alloc{
      .size = this->used_allocs.at( ptr ),
      .ptr = ptr
    };

    this->used_allocs.erase( ptr );
    this->free_allocs.insert( alloc );
  }
};

template <typename T, typename... Ts>
struct unique : std::type_identity<T> {};

template <typename... Ts, typename U, typename... Us>
struct unique<std::tuple<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
                       , unique<std::tuple<Ts...>, Us...>
                       , unique<std::tuple<Ts..., U>, Us...>> {};

template <typename... Ts>
using unique_tuple = typename unique<std::tuple<>, Ts...>::type;

} // MpiBufferPool_Impl

class MpiBufferPool{
private:  
  using HostMemSpace = Kokkos::DefaultHostExecutionSpace::memory_space;
  using DefaultMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  template< typename memory_space_t >
  using pool_t = MpiBufferPool_Impl::pool_t<memory_space_t>;

  MpiBufferPool_Impl::unique_tuple< 
    pool_t<HostMemSpace>,
    pool_t<DefaultMemSpace>
  > pools; /// Separate pools for each memory_space
  // Note : unique_tuple is needed in case HostMemSpace == DefaultMemSpace

  template< typename memory_space_t >
  auto& getPool()
  {
    return std::get<pool_t<memory_space_t>>(this->pools);
  }

public:
  ~MpiBufferPool()
  {
    this->clear();
  }  

  void clear()
  {
    std::apply(
      [](auto&&... args) 
        {((args.clear()), ...);},
      this->pools );   
  }  

  template< typename View_t, typename... IntType >
  View_t MPI_Alloc_view( const std::string& name, const IntType&... extents )
  {
    using value_type = typename View_t::value_type;
    using memory_space = typename View_t::memory_space;

    size_t req_size = ((extents * ...)) * sizeof( value_type );    
    
    if(req_size > 0)
    {
      void* ptr = this->getPool<memory_space>().alloc(req_size);
      return View_t( static_cast<value_type*>(ptr), extents... );
    }
    else 
      return View_t( "name", extents... );
  }

  template< typename View_t >
  void MPI_Free_view( View_t& view )
  {
    if( view.size() > 0 )
    {
      void* ptr = view.data();
      using memory_space = typename View_t::memory_space;
      this->getPool<memory_space>().free(ptr);
    }
    view = View_t();
  }
};

}